import os
import math
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, session, redirect, render_template, abort

# -----------------------
# Config (env vars)
# -----------------------
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
OANDA_API_KEY = os.environ.get("OANDA_API_KEY", "")
OANDA_BASE_URL = os.environ.get("OANDA_BASE_URL", "https://api-fxtrade.oanda.com")
OANDA_INSTRUMENT = os.environ.get("OANDA_INSTRUMENT", "XAU_USD")
APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "Asia/Singapore")

RR = float(os.environ.get("RR", "2.0"))
DEFAULT_EQUITY = float(os.environ.get("DEFAULT_EQUITY", "500"))

# Your tick info (per 1.00 lot): tick size 0.01, tick value $1
TICK_SIZE = float(os.environ.get("XAUUSD_TICK_SIZE", "0.01"))
TICK_VALUE = float(os.environ.get("XAUUSD_TICK_VALUE", "1.0"))

# Derived: $ per $1 move per lot = (1 / tick_size) * tick_value
DOLLARS_PER_1USD_MOVE_PER_LOT = (1.0 / TICK_SIZE) * TICK_VALUE  # = 100 when tick_size=0.01 and tick_value=1

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(32))

def sgt(dt: datetime) -> str:
    tz = ZoneInfo(APP_TIMEZONE)
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")

def oanda_headers():
    if not OANDA_API_KEY:
        raise RuntimeError("Missing OANDA_API_KEY")
    return {"Authorization": f"Bearer {OANDA_API_KEY}"}

def fetch_candles(granularity: str, count: int = 500):
    url = f"{OANDA_BASE_URL}/v3/instruments/{OANDA_INSTRUMENT}/candles"
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = requests.get(url, headers=oanda_headers(), params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["candles"]
    rows = []
    for c in data:
        if not c.get("complete"):
            continue
        t = datetime.fromisoformat(c["time"].replace("Z", "+00:00"))
        m = c["mid"]
        rows.append({
            "time": t,
            "o": float(m["o"]),
            "h": float(m["h"]),
            "l": float(m["l"]),
            "c": float(m["c"]),
        })
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    prev_close = df["c"].shift(1)
    tr = pd.concat([
        (df["h"] - df["l"]),
        (df["h"] - prev_close).abs(),
        (df["l"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def session_tag(dt_utc: datetime) -> str:
    # Rough session buckets in SGT (not DST-perfect, but good enough for MVP)
    # SGT = UTC+8
    h = dt_utc.astimezone(ZoneInfo(APP_TIMEZONE)).hour
    if 7 <= h < 15:
        return "ASIA"
    if 15 <= h < 20:
        return "LONDON"
    if 20 <= h or h < 2:
        return "NY"
    return "OFF"

def swing_levels(df, lookback=20):
    # simple swing high/low
    hi = df["h"].rolling(lookback).max()
    lo = df["l"].rolling(lookback).min()
    return hi, lo

def compute_signal():
    df15 = fetch_candles("M15", 400)
    df1h = fetch_candles("H1", 400)
    df4h = fetch_candles("H4", 400)

    # Indicators
    df15["ema50"] = ema(df15["c"], 50)
    df15["ema200"] = ema(df15["c"], 200)
    df15["atr14"] = atr(df15, 14)

    df1h["ema50"] = ema(df1h["c"], 50)
    df1h["ema200"] = ema(df1h["c"], 200)

    df4h["ema50"] = ema(df4h["c"], 50)
    df4h["ema200"] = ema(df4h["c"], 200)

    # Bias
    bull = (df1h["ema50"].iloc[-1] > df1h["ema200"].iloc[-1]) and (df4h["ema50"].iloc[-1] > df4h["ema200"].iloc[-1])
    bear = (df1h["ema50"].iloc[-1] < df1h["ema200"].iloc[-1]) and (df4h["ema50"].iloc[-1] < df4h["ema200"].iloc[-1])
    bias = "NEUTRAL"
    if bull:
        bias = "BULL"
    elif bear:
        bias = "BEAR"

    # S/R proxy via swings
    hi, lo = swing_levels(df15, 20)
    last = df15.iloc[-1]
    prev = df15.iloc[-2]
    price = float(last["c"])
    a = float(last["atr14"]) if not math.isnan(last["atr14"]) else 2.0

    # Simple SMC-like triggers:
    # Sweep + reclaim:
    swept_low = (prev["l"] < lo.iloc[-2]) and (last["c"] > lo.iloc[-2])
    swept_high = (prev["h"] > hi.iloc[-2]) and (last["c"] < hi.iloc[-2])

    # Momentum confirmation:
    mom_up = last["c"] > last["ema50"]
    mom_dn = last["c"] < last["ema50"]

    # "ML score" MVP (placeholder scoring; later replaced with trained model)
    # Score based on confluence; 0-1.
    score = 0.5
    reason = []
    if bias == "BULL":
        score += 0.12; reason.append("HTF_BULL")
    if bias == "BEAR":
        score += 0.12; reason.append("HTF_BEAR")
    if swept_low:
        score += 0.18; reason.append("SWEEP_LOW_RECLAIM")
    if swept_high:
        score += 0.18; reason.append("SWEEP_HIGH_REJECT")
    if mom_up:
        score += 0.08; reason.append("ABOVE_EMA50")
    if mom_dn:
        score += 0.08; reason.append("BELOW_EMA50")

    # Session thresholding
    now_utc = df15.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    sess = session_tag(now_utc)
    thresh = 0.62 if sess == "ASIA" else 0.58  # stricter in Asia
    reason.append(f"SESSION_{sess}")

    side = "NO_TRADE"
    entry = sl = tp = None

    # Entry rules
    if swept_low and (bias in ["BULL", "NEUTRAL"]) and mom_up and score >= thresh:
        side = "BUY"
        entry = price
        base_sl = float(min(df15["l"].iloc[-6:]))  # recent low
        sl = base_sl - (a * 0.4)
        tp = entry + RR * (entry - sl)
    elif swept_high and (bias in ["BEAR", "NEUTRAL"]) and mom_dn and score >= thresh:
        side = "SELL"
        entry = price
        base_sl = float(max(df15["h"].iloc[-6:]))  # recent high
        sl = base_sl + (a * 0.4)
        tp = entry - RR * (sl - entry)

    out = {
        "time_utc": df15.index[-1].to_pydatetime().replace(tzinfo=timezone.utc),
        "time_sgt": sgt(df15.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)),
        "symbol": "XAUUSD",
        "price": round(price, 2),
        "side": side,
        "entry": round(entry, 2) if entry else "-",
        "sl": round(sl, 2) if sl else "-",
        "tp": round(tp, 2) if tp else "-",
        "rr": RR,
        "ml_score": round(float(min(max(score, 0.0), 1.0)), 2),
        "session": sess,
        "reason": ", ".join(reason[:6]),
    }
    return out

def compute_lots(equity: float, risk_pct: float, entry, sl) -> str:
    if entry == "-" or sl == "-":
        return "-"
    risk_usd = equity * (risk_pct / 100.0)
    sl_dist = abs(float(entry) - float(sl))
    if sl_dist <= 0:
        return "-"
    loss_per_lot = sl_dist * DOLLARS_PER_1USD_MOVE_PER_LOT
    lots = risk_usd / loss_per_lot
    # Pepperstone min step 0.01
    lots = math.floor(lots / 0.01) * 0.01
    return f"{max(lots, 0.0):.2f}"

# -----------------------
# Auth (simple shared password)
# -----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pw = request.form.get("password", "")
        if APP_PASSWORD and pw == APP_PASSWORD:
            session["authed"] = True
            return redirect("/")
        return "Wrong password", 401
    return """
    <html><body style="font-family:Arial;background:#0b0f17;color:#e8eefc;display:flex;justify-content:center;align-items:center;height:100vh;">
      <form method="post" style="background:#111a2e;border:1px solid #1f2a44;padding:18px;border-radius:12px;min-width:320px;">
        <h3>XAUUSD Signals</h3>
        <p>Enter password</p>
        <input name="password" type="password" style="width:100%;padding:10px;border-radius:10px;border:1px solid #1f2a44;background:#0b0f17;color:#e8eefc;" />
        <button style="margin-top:12px;width:100%;padding:10px;border-radius:10px;border:0;background:#3b82f6;color:white;font-weight:700;">Login</button>
      </form>
    </body></html>
    """

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/login")

@app.route("/")
def home():
    if not session.get("authed"):
        return redirect("/login")

    equity = float(request.args.get("equity", DEFAULT_EQUITY))
    risk = int(request.args.get("risk", "1"))

    # cache in memory for 60s to avoid hammering OANDA
    now = time.time()
    cache = app.config.get("CACHE")
    if not cache or now - cache["ts"] > 60:
        sig = compute_signal()
        hist = app.config.get("HISTORY", [])
        hist = ([sig] + hist)[:30]
        app.config["HISTORY"] = hist
        app.config["CACHE"] = {"ts": now, "sig": sig}
    else:
        sig = cache["sig"]
        hist = app.config.get("HISTORY", [sig])

    lots = compute_lots(equity, risk, sig["entry"], sig["sl"])
    return render_template("index.html", signal=sig, history=hist, equity=int(equity), risk=risk, lots=lots)

@app.get("/health")
def health():
    return {"ok": True}