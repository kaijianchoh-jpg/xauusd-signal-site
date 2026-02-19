import os
import math
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, request, session, redirect, render_template

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

# Tick info (per 1.00 lot): tick size 0.01, tick value $1
TICK_SIZE = float(os.environ.get("XAUUSD_TICK_SIZE", "0.01"))
TICK_VALUE = float(os.environ.get("XAUUSD_TICK_VALUE", "1.0"))
DOLLARS_PER_1USD_MOVE_PER_LOT = (1.0 / TICK_SIZE) * TICK_VALUE  # 100 when 0.01 & 1.0

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
NOTIFY_ALL = os.environ.get("NOTIFY_ALL", "false").lower() == "true"
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "3"))

# Auto-update trigger (UptimeRobot will call this)
PING_KEY = os.environ.get("PING_KEY", "")

# NO_TRADE plan-change threshold (gold dollars)
NO_TRADE_CHANGE_THRESHOLD = 0.30

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(32))

def sgt(dt: datetime) -> str:
    tz = ZoneInfo(APP_TIMEZONE)
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")

def oanda_headers():
    if not OANDA_API_KEY:
        raise RuntimeError("Missing OANDA_API_KEY (set it in Render Environment)")
    return {"Authorization": f"Bearer {OANDA_API_KEY}"}

def fetch_candles(granularity: str, count: int = 500) -> pd.DataFrame:
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
        rows.append(
            {"time": t, "o": float(m["o"]), "h": float(m["h"]), "l": float(m["l"]), "c": float(m["c"])}
        )

    return pd.DataFrame(rows).set_index("time").sort_index()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["c"].shift(1)
    tr = pd.concat(
        [(df["h"] - df["l"]), (df["h"] - prev_close).abs(), (df["l"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()

def session_tag(dt_utc: datetime) -> str:
    h = dt_utc.astimezone(ZoneInfo(APP_TIMEZONE)).hour
    if 7 <= h < 15:
        return "ASIA"
    if 15 <= h < 20:
        return "LONDON"
    if 20 <= h or h < 2:
        return "NY"
    return "OFF"

def swing_levels(df: pd.DataFrame, lookback: int = 20):
    hi = df["h"].rolling(lookback).max()
    lo = df["l"].rolling(lookback).min()
    return hi, lo

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code >= 400:
            app.logger.error("Telegram send failed: %s %s", r.status_code, r.text[:300])
    except Exception as e:
        app.logger.error("Telegram error: %s", str(e))

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

    # HTF bias
    bull = (df1h["ema50"].iloc[-1] > df1h["ema200"].iloc[-1]) and (
        df4h["ema50"].iloc[-1] > df4h["ema200"].iloc[-1]
    )
    bear = (df1h["ema50"].iloc[-1] < df1h["ema200"].iloc[-1]) and (
        df4h["ema50"].iloc[-1] < df4h["ema200"].iloc[-1]
    )
    bias = "NEUTRAL"
    if bull:
        bias = "BULL"
    elif bear:
        bias = "BEAR"

    # Structure proxy
    hi, lo = swing_levels(df15, 20)
    last = df15.iloc[-1]
    prev = df15.iloc[-2]
    price = float(last["c"])
    a = float(last["atr14"]) if not math.isnan(last["atr14"]) else 2.0

    # Sweep + reclaim/reject
    swept_low = (prev["l"] < lo.iloc[-2]) and (last["c"] > lo.iloc[-2])
    swept_high = (prev["h"] > hi.iloc[-2]) and (last["c"] < hi.iloc[-2])

    mom_up = last["c"] > last["ema50"]
    mom_dn = last["c"] < last["ema50"]

    # MVP "score" (confluence score)
    score = 0.5
    reason = []
    if bias == "BULL":
        score += 0.12
        reason.append("HTF_BULL")
    if bias == "BEAR":
        score += 0.12
        reason.append("HTF_BEAR")
    if swept_low:
        score += 0.18
        reason.append("SWEEP_LOW_RECLAIM")
    if swept_high:
        score += 0.18
        reason.append("SWEEP_HIGH_REJECT")
    if mom_up:
        score += 0.08
        reason.append("ABOVE_EMA50")
    if mom_dn:
        score += 0.08
        reason.append("BELOW_EMA50")

    now_utc = df15.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
    sess = session_tag(now_utc)
    reason.append(f"SESSION_{sess}")

    thresh = 0.62 if sess == "ASIA" else 0.58

    side = "NO_TRADE"
    entry = sl = tp = None

    # Confirmed BUY/SELL
    if swept_low and (bias in ["BULL", "NEUTRAL"]) and mom_up and score >= thresh:
        side = "BUY"
        entry = price
        base_sl = float(min(df15["l"].iloc[-6:]))
        sl = base_sl - (a * 0.4)
        tp = entry + RR * (entry - sl)

    elif swept_high and (bias in ["BEAR", "NEUTRAL"]) and mom_dn and score >= thresh:
        side = "SELL"
        entry = price
        base_sl = float(max(df15["h"].iloc[-6:]))
        sl = base_sl + (a * 0.4)
        tp = entry - RR * (sl - entry)

    # POTENTIAL plan even if NO_TRADE
    if side == "NO_TRADE":
        entry = price
        if bias == "BULL":
            base_sl = float(min(df15["l"].iloc[-6:]))
            sl = base_sl - (a * 0.4)
            tp = entry + RR * (entry - sl)
            reason.append("POTENTIAL_BUY_PLAN")
        elif bias == "BEAR":
            base_sl = float(max(df15["h"].iloc[-6:]))
            sl = base_sl + (a * 0.4)
            tp = entry - RR * (sl - entry)
            reason.append("POTENTIAL_SELL_PLAN")
        else:
            sl = entry - (a * 1.0)
            tp = entry + RR * (entry - sl)
            reason.append("POTENTIAL_NEUTRAL_PLAN")

    out = {
        "time_utc": now_utc,
        "time_sgt": sgt(now_utc),
        "symbol": "XAUUSD",
        "price": round(price, 2),
        "side": side,
        "entry": round(float(entry), 2) if entry is not None else "-",
        "sl": round(float(sl), 2) if sl is not None else "-",
        "tp": round(float(tp), 2) if tp is not None else "-",
        "rr": RR,
        "ml_score": round(float(min(max(score, 0.0), 1.0)), 2),
        "session": sess,
        "reason": ", ".join(reason[:10]),
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
    lots = math.floor(lots / 0.01) * 0.01
    return f"{max(lots, 0.0):.2f}"

def maybe_notify(sig: dict):
    # Cooldown by side
    last = app.config.get("LAST_NOTIFY", {})
    now_ts = time.time()
    last_ts = last.get(sig["side"], 0)
    if (now_ts - last_ts) < COOLDOWN_MINUTES * 60:
        return

    # If NO_TRADE, only notify if plan changed enough
    if sig["side"] == "NO_TRADE":
        prev = app.config.get("PREV_PLAN", {})
        price_change = abs(float(sig["price"]) - float(prev.get("price", sig["price"])))
        entry_change = abs(float(sig["entry"]) - float(prev.get("entry", sig["entry"])))
        sl_change = abs(float(sig["sl"]) - float(prev.get("sl", sig["sl"])))
        tp_change = abs(float(sig["tp"]) - float(prev.get("tp", sig["tp"])))

        if max(price_change, entry_change, sl_change, tp_change) < NO_TRADE_CHANGE_THRESHOLD:
            last[sig["side"]] = now_ts
            app.config["LAST_NOTIFY"] = last
            return

        app.config["PREV_PLAN"] = {"price": sig["price"], "entry": sig["entry"], "sl": sig["sl"], "tp": sig["tp"]}

    if NOTIFY_ALL or sig["side"] in ["BUY", "SELL"]:
        msg = (
            f"XAUUSD {sig.get('type','')}".strip() + f" {sig['side']} (15M)\n"
            f"Time(SGT): {sig['time_sgt']}\n"
            f"Price: {sig['price']}\n"
            f"Entry: {sig['entry']}  SL: {sig['sl']}  TP: {sig['tp']}  RR: {sig['rr']}\n"
            f"Session: {sig['session']}  Score: {sig['ml_score']}\n"
            f"Reason: {sig['reason']}"
        )
        send_telegram(msg)

    last[sig["side"]] = now_ts
    app.config["LAST_NOTIFY"] = last

# -----------------------
# Auth (shared password)
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

# -----------------------
# UptimeRobot trigger
# -----------------------
@app.get("/ping/run")
def ping_run():
    key = request.args.get("key", "")
    if not PING_KEY or key != PING_KEY:
        return {"ok": False, "error": "unauthorized"}, 401

    sig = compute_signal()
    sig["type"] = "AUTO"

    maybe_notify(sig)

    # Update cache/history for the website
    app.config["CACHE"] = {"ts": time.time(), "sig": sig}
    hist = app.config.get("HISTORY", [])
    hist = ([sig] + hist)[:60]
    app.config["HISTORY"] = hist

    return {"ok": True, "side": sig["side"], "time_sgt": sig["time_sgt"]}

@app.route("/")
def home():
    if not session.get("authed"):
        return redirect("/login")

    equity = float(request.args.get("equity", DEFAULT_EQUITY))
    risk = int(request.args.get("risk", "1"))

    cache = app.config.get("CACHE")
    if not cache:
        sig = compute_signal()
        sig["type"] = "ON_DEMAND"
        app.config["CACHE"] = {"ts": time.time(), "sig": sig}
        hist = app.config.get("HISTORY", [])
        hist = ([sig] + hist)[:60]
        app.config["HISTORY"] = hist
    else:
        sig = cache["sig"]
        hist = app.config.get("HISTORY", [sig])

    lots = compute_lots(equity, risk, sig["entry"], sig["sl"])
    return render_template("index.html", signal=sig, history=hist, equity=int(equity), risk=risk, lots=lots)

@app.get("/health")
def health():
    return {"ok": True}
