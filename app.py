"""
Advanced Multi-Strategy SMC Scanner v2
Strategies: SMC + ICT + Technical (RSI/MACD/EMA) + Price Action
AI Confidence Scoring: 0-100% based on multi-strategy confluence
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz, os

app = Flask(__name__, static_folder="static")
CORS(app)

SYMBOLS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X", "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X", "AUDJPY": "AUDJPY=X", "EURAUD": "EURAUD=X",
    "US30": "^DJI", "BTCUSD": "BTC-USD", "XAUUSD": "GC=F",
}

TIMEFRAMES = {
    "M5":  {"interval": "5m",  "period": "5d"},
    "M15": {"interval": "15m", "period": "5d"},
    "M30": {"interval": "30m", "period": "7d"},
    "H1":  {"interval": "1h",  "period": "30d"},
    "H4":  {"interval": "4h",  "period": "60d"},
    "D1":  {"interval": "1d",  "period": "180d"},
}

# ICT killzone hours (UTC)
KILLZONES = {
    "London":    (7, 10),
    "New York":  (12, 15),
    "Asian":     (0, 3),
    "London/NY": (12, 13),
}

SWING_LB = 10
SAST = pytz.timezone("Africa/Johannesburg")


# ── Data ─────────────────────────────────────────────────────────────
def get_data(ticker, interval, period):
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         auto_adjust=True, progress=False, silence_errors=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except:
        return pd.DataFrame()


def dp(symbol):
    if symbol == "BTCUSD": return 2
    if symbol == "US30":   return 1
    if symbol == "XAUUSD": return 2
    if "JPY" in symbol:    return 3
    return 5


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 1 — SMC
# ══════════════════════════════════════════════════════════════════════
def smc_analysis(df, symbol):
    if len(df) < SWING_LB + 5: return None
    h, l, o, c = df["High"], df["Low"], df["Open"], df["Close"]

    sh = h.iloc[-(SWING_LB+2):-1].max()
    sl = l.iloc[-(SWING_LB+2):-1].min()
    last = c.iloc[-1]
    signals = []

    # BOS
    if last > sh:
        signals.append({"name": "BOS", "direction": "Bullish", "strength": 0.7})
    if last < sl:
        signals.append({"name": "BOS", "direction": "Bearish", "strength": 0.7})

    # CHoCH
    dn = h.iloc[-4] > h.iloc[-3] > h.iloc[-2] and l.iloc[-4] > l.iloc[-3] > l.iloc[-2]
    up = h.iloc[-4] < h.iloc[-3] < h.iloc[-2] and l.iloc[-4] < l.iloc[-3] < l.iloc[-2]
    if dn and last > h.iloc[-2]:
        signals.append({"name": "CHoCH", "direction": "Bullish", "strength": 0.9})
    if up and last < l.iloc[-2]:
        signals.append({"name": "CHoCH", "direction": "Bearish", "strength": 0.9})

    # FVG
    bull_gap = l.iloc[-1] - h.iloc[-3]
    bear_gap = l.iloc[-3] - h.iloc[-1]
    thresh   = last * 0.0002
    if bull_gap > thresh:
        signals.append({"name": "FVG", "direction": "Bullish", "strength": 0.65})
    if bear_gap > thresh:
        signals.append({"name": "FVG", "direction": "Bearish", "strength": 0.65})

    # OB — last opposing candle near price
    spread = last * 0.001
    if c.iloc[-1] > o.iloc[-1]:
        for i in range(2, SWING_LB):
            if c.iloc[-i] < o.iloc[-i] and abs(last - h.iloc[-i]) < spread:
                signals.append({"name": "OB", "direction": "Bullish", "strength": 0.75})
                break
    if c.iloc[-1] < o.iloc[-1]:
        for i in range(2, SWING_LB):
            if c.iloc[-i] > o.iloc[-i] and abs(last - l.iloc[-i]) < spread:
                signals.append({"name": "OB", "direction": "Bearish", "strength": 0.75})
                break

    return signals if signals else None


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 2 — ICT
# ══════════════════════════════════════════════════════════════════════
def ict_analysis(df, symbol):
    if len(df) < 20: return None
    h, l, o, c = df["High"], df["Low"], df["Open"], df["Close"]
    signals = []

    # Killzone detection
    now_utc   = datetime.utcnow().hour
    in_kz     = None
    for name, (start, end) in KILLZONES.items():
        if start <= now_utc < end:
            in_kz = name
            break
    kz_bonus = 0.15 if in_kz else 0

    # PD Array — Premium / Discount
    recent_high = h.iloc[-20:].max()
    recent_low  = l.iloc[-20:].min()
    midpoint    = (recent_high + recent_low) / 2
    last        = c.iloc[-1]

    # Discount = below midpoint (look for buys), Premium = above (look for sells)
    in_discount = last < midpoint
    in_premium  = last > midpoint

    # Optimal Trade Entry (OTE) — 0.62–0.79 Fibonacci retracement
    fib_range = recent_high - recent_low
    ote_low   = recent_high - fib_range * 0.79
    ote_high  = recent_high - fib_range * 0.62

    if in_discount and ote_low <= last <= ote_high:
        signals.append({"name": "ICT OTE", "direction": "Bullish",
                         "strength": 0.8 + kz_bonus,
                         "killzone": in_kz or "–"})
    if in_premium and ote_low <= last <= ote_high:
        signals.append({"name": "ICT OTE", "direction": "Bearish",
                         "strength": 0.8 + kz_bonus,
                         "killzone": in_kz or "–"})

    # IPDA swing high/low (20-bar)
    swing_h20 = h.iloc[-20:].max()
    swing_l20 = l.iloc[-20:].min()
    if last > swing_h20 * 0.999:
        signals.append({"name": "IPDA High", "direction": "Bearish", "strength": 0.6 + kz_bonus})
    if last < swing_l20 * 1.001:
        signals.append({"name": "IPDA Low",  "direction": "Bullish", "strength": 0.6 + kz_bonus})

    return signals if signals else None


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Technical Indicators
# ══════════════════════════════════════════════════════════════════════
def technical_analysis(df, symbol):
    if len(df) < 30: return None
    c = df["Close"]
    signals = []

    # EMA 8 / 21 / 50 crossover
    ema8  = c.ewm(span=8,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    bull_ema = ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
    bear_ema = ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]
    # Fresh cross (previous bar was opposite)
    bull_cross = ema8.iloc[-2] <= ema21.iloc[-2] and ema8.iloc[-1] > ema21.iloc[-1]
    bear_cross = ema8.iloc[-2] >= ema21.iloc[-2] and ema8.iloc[-1] < ema21.iloc[-1]

    if bull_cross:
        signals.append({"name": "EMA Cross ↑", "direction": "Bullish", "strength": 0.75})
    elif bull_ema:
        signals.append({"name": "EMA Stack ↑", "direction": "Bullish", "strength": 0.55})
    if bear_cross:
        signals.append({"name": "EMA Cross ↓", "direction": "Bearish", "strength": 0.75})
    elif bear_ema:
        signals.append({"name": "EMA Stack ↓", "direction": "Bearish", "strength": 0.55})

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    rsi_v = rsi.iloc[-1]

    if rsi_v < 35:
        signals.append({"name": f"RSI OS ({rsi_v:.0f})", "direction": "Bullish",
                         "strength": 0.7 if rsi_v < 30 else 0.55})
    if rsi_v > 65:
        signals.append({"name": f"RSI OB ({rsi_v:.0f})", "direction": "Bearish",
                         "strength": 0.7 if rsi_v > 70 else 0.55})

    # MACD
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal

    macd_bull = hist.iloc[-1] > 0 and hist.iloc[-2] <= 0
    macd_bear = hist.iloc[-1] < 0 and hist.iloc[-2] >= 0

    if macd_bull:
        signals.append({"name": "MACD Cross ↑", "direction": "Bullish", "strength": 0.7})
    if macd_bear:
        signals.append({"name": "MACD Cross ↓", "direction": "Bearish", "strength": 0.7})

    return signals if signals else None


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 4 — Price Action
# ══════════════════════════════════════════════════════════════════════
def price_action_analysis(df, symbol):
    if len(df) < 5: return None
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    signals = []

    body    = abs(c.iloc[-1] - o.iloc[-1])
    candle  = h.iloc[-1] - l.iloc[-1]
    upper_w = h.iloc[-1] - max(c.iloc[-1], o.iloc[-1])
    lower_w = min(c.iloc[-1], o.iloc[-1]) - l.iloc[-1]
    is_bull  = c.iloc[-1] > o.iloc[-1]
    is_bear  = c.iloc[-1] < o.iloc[-1]

    # Pin bar / hammer
    if candle > 0:
        if lower_w > body * 2 and upper_w < body * 0.5:
            signals.append({"name": "Hammer", "direction": "Bullish", "strength": 0.7})
        if upper_w > body * 2 and lower_w < body * 0.5:
            signals.append({"name": "Shooting Star", "direction": "Bearish", "strength": 0.7})

    # Engulfing
    prev_body = abs(c.iloc[-2] - o.iloc[-2])
    if is_bull and c.iloc[-2] < o.iloc[-2] and body > prev_body * 1.1:
        signals.append({"name": "Bull Engulf", "direction": "Bullish", "strength": 0.75})
    if is_bear and c.iloc[-2] > o.iloc[-2] and body > prev_body * 1.1:
        signals.append({"name": "Bear Engulf", "direction": "Bearish", "strength": 0.75})

    # Inside bar (consolidation breakout setup)
    if h.iloc[-1] < h.iloc[-2] and l.iloc[-1] > l.iloc[-2]:
        signals.append({"name": "Inside Bar", "direction": "Bullish" if is_bull else "Bearish",
                         "strength": 0.55})

    # Doji — indecision at key level
    if candle > 0 and body / candle < 0.1:
        signals.append({"name": "Doji", "direction": "Bullish" if is_bull else "Bearish",
                         "strength": 0.45})

    # Morning / Evening star (3-candle)
    if len(df) >= 3:
        doji_mid = abs(c.iloc[-2] - o.iloc[-2]) / max(h.iloc[-2] - l.iloc[-2], 0.0001) < 0.2
        if c.iloc[-3] < o.iloc[-3] and doji_mid and c.iloc[-1] > o.iloc[-1]:
            signals.append({"name": "Morning Star", "direction": "Bullish", "strength": 0.85})
        if c.iloc[-3] > o.iloc[-3] and doji_mid and c.iloc[-1] < o.iloc[-1]:
            signals.append({"name": "Evening Star", "direction": "Bearish", "strength": 0.85})

    return signals if signals else None


# ══════════════════════════════════════════════════════════════════════
# AI CONFIDENCE SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════
def ai_score(smc, ict, tech, pa, direction):
    """
    Weighted confluence scoring.
    Each strategy contributes to a 0-100 confidence score.
    Weights: SMC 35% | ICT 30% | Technical 20% | PA 15%
    Bonus: +5 for each extra confirming signal, +10 for killzone
    """
    weights   = {"smc": 0.35, "ict": 0.30, "tech": 0.20, "pa": 0.15}
    score     = 0.0
    confirmed = 0
    tags      = []
    in_kz     = None

    def best(signals, w):
        nonlocal confirmed, in_kz
        if not signals: return 0
        matching = [s for s in signals if s["direction"] == direction]
        if not matching: return 0
        confirmed += 1
        best_s = max(matching, key=lambda x: x["strength"])
        tags.append(best_s["name"])
        if "killzone" in best_s and best_s["killzone"] != "–":
            in_kz = best_s["killzone"]
        return best_s["strength"] * w * 100

    score += best(smc,  weights["smc"])
    score += best(ict,  weights["ict"])
    score += best(tech, weights["tech"])
    score += best(pa,   weights["pa"])

    # Confluence bonus
    score += (confirmed - 1) * 5 if confirmed > 1 else 0

    # Killzone bonus
    if in_kz:
        score += 10
        tags.append(f"{in_kz} KZ")

    score = min(round(score), 99)  # cap at 99 — never 100%

    # Grade label
    if score >= 80:   grade = "A"
    elif score >= 65: grade = "B"
    elif score >= 50: grade = "C"
    else:             grade = "D"

    return score, grade, confirmed, tags, in_kz


# ══════════════════════════════════════════════════════════════════════
# SIGNAL BUILDER
# ══════════════════════════════════════════════════════════════════════
def build_signal(df, symbol, tf_name, direction, score, grade,
                 confirmed, tags, in_kz):
    h, l, c = df["High"], df["Low"], df["Close"]
    last = float(c.iloc[-1])
    buf  = last * 0.0003

    # Find OB for SL anchor
    ob_h, ob_l = last * 1.002, last * 0.998
    for i in range(2, SWING_LB):
        if direction == "Bullish" and c.iloc[-i] < df["Open"].iloc[-i]:
            ob_h, ob_l = float(h.iloc[-i]), float(l.iloc[-i])
            break
        if direction == "Bearish" and c.iloc[-i] > df["Open"].iloc[-i]:
            ob_h, ob_l = float(h.iloc[-i]), float(l.iloc[-i])
            break

    d = dp(symbol)
    if direction == "Bullish":
        sl   = ob_l - buf
        risk = last - sl
        if risk <= 0: return None
        tp1  = last + risk
        # TP2: next swing high
        candidates = [float(h.iloc[-i]) for i in range(1, SWING_LB*2)
                      if i+1 < len(h) and float(h.iloc[-i]) > last]
        tp2  = min(candidates) if candidates else last + risk * 2
        rr   = round((tp2 - last) / risk, 2)
    else:
        sl   = ob_h + buf
        risk = sl - last
        if risk <= 0: return None
        tp1  = last - risk
        candidates = [float(l.iloc[-i]) for i in range(1, SWING_LB*2)
                      if i+1 < len(l) and float(l.iloc[-i]) < last]
        tp2  = max(candidates) if candidates else last - risk * 2
        rr   = round((last - tp2) / risk, 2)

    if rr < 1.0: return None

    return {
        "symbol":    symbol,
        "timeframe": tf_name,
        "direction": direction,
        "score":     score,
        "grade":     grade,
        "confirmed": confirmed,
        "tags":      tags,
        "killzone":  in_kz,
        "entry":     round(last, d),
        "sl":        round(sl,   d),
        "tp1":       round(tp1,  d),
        "tp2":       round(tp2,  d),
        "rr":        rr,
        "time":      datetime.now(SAST).strftime("%H:%M SAST"),
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ══════════════════════════════════════════════════════════════════════
def run_scanner(min_score=30):
    signals = []
    for display, ticker in SYMBOLS.items():
        for tf_name, tf_cfg in TIMEFRAMES.items():
            df = get_data(ticker, tf_cfg["interval"], tf_cfg["period"])
            if df.empty or len(df) < 30:
                continue

            smc  = smc_analysis(df, display)
            ict  = ict_analysis(df, display)
            tech = technical_analysis(df, display)
            pa   = price_action_analysis(df, display)

            # Try both directions
            for direction in ["Bullish", "Bearish"]:
                score, grade, confirmed, tags, in_kz = \
                    ai_score(smc, ict, tech, pa, direction)

                if score >= min_score and confirmed >= 1:
                    sig = build_signal(df, display, tf_name, direction,
                                       score, grade, confirmed, tags, in_kz)
                    if sig:
                        signals.append(sig)

    # Sort by score descending
    signals.sort(key=lambda x: -x["score"])
    return signals


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/signals")
def api_signals():
    min_score = 30
    signals   = run_scanner(min_score)
    now       = datetime.now(SAST).strftime("%H:%M:%S SAST")
    return jsonify({
        "signals":   signals,
        "count":     len(signals),
        "scanned":   len(SYMBOLS) * len(TIMEFRAMES),
        "updated":   now,
        "min_score": min_score,
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "2.0"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
