"""
SMC Market Scanner — Python Backend
Fetches OHLCV data, runs SMC analysis, serves signals via REST API
"""

from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

app = Flask(__name__)
CORS(app)

# ── Symbols ──────────────────────────────────────────────────────────
SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
    "AUDJPY": "AUDJPY=X",
    "EURAUD": "EURAUD=X",
    "US30":   "^DJI",
    "BTCUSD": "BTC-USD",
}

# ── Timeframes (yfinance interval : period to fetch) ──────────────────
TIMEFRAMES = {
    "M5":  {"interval": "5m",  "period": "5d"},
    "M15": {"interval": "15m", "period": "5d"},
    "M30": {"interval": "30m", "period": "5d"},
    "H1":  {"interval": "1h",  "period": "30d"},
    "H4":  {"interval": "4h",  "period": "60d"},
    "D1":  {"interval": "1d",  "period": "180d"},
}

SWING_LOOKBACK = 10
MIN_RR         = 1.0   # skip signals below this R:R


# ── Data fetcher ─────────────────────────────────────────────────────
def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


# ── SMC helpers ──────────────────────────────────────────────────────
def swing_high_low(high: pd.Series, low: pd.Series, lookback: int):
    """Return the highest high and lowest low over the lookback window."""
    sh = high.iloc[-lookback:].max()
    sl = low.iloc[-lookback:].min()
    sh_idx = int(high.iloc[-lookback:].argmax())
    sl_idx = int(low.iloc[-lookback:].argmin())
    return sh, sl, sh_idx, sl_idx


def find_ob(open_: pd.Series, close_: pd.Series,
            high_: pd.Series, low_: pd.Series,
            direction: str, lookback: int):
    """Find the last opposing candle (Order Block) before current move."""
    for i in range(2, lookback):
        idx = -(i + 1)
        if direction == "Bullish" and close_.iloc[idx] < open_.iloc[idx]:
            return high_.iloc[idx], low_.iloc[idx]
        if direction == "Bearish" and close_.iloc[idx] > open_.iloc[idx]:
            return high_.iloc[idx], low_.iloc[idx]
    return None, None


def find_next_liquidity(high_: pd.Series, low_: pd.Series,
                        direction: str, entry: float, lookback: int):
    """Nearest swing high/low beyond entry for TP2."""
    if direction == "Bullish":
        candidates = []
        for i in range(1, lookback * 2):
            if i + 1 >= len(high_): break
            if high_.iloc[-i] > high_.iloc[-(i+1)] and \
               high_.iloc[-i] > high_.iloc[-(i-1)] and \
               high_.iloc[-i] > entry:
                candidates.append(high_.iloc[-i])
        return min(candidates) if candidates else entry * 1.002
    else:
        candidates = []
        for i in range(1, lookback * 2):
            if i + 1 >= len(low_): break
            if low_.iloc[-i] < low_.iloc[-(i+1)] and \
               low_.iloc[-i] < low_.iloc[-(i-1)] and \
               low_.iloc[-i] < entry:
                candidates.append(low_.iloc[-i])
        return max(candidates) if candidates else entry * 0.998


def build_signal(symbol, tf_name, sig_type, direction,
                 ob_high, ob_low, last_close,
                 high_, low_, buffer_pct=0.0003):
    """Calculate entry, SL, TP1, TP2, R:R and return signal dict."""
    entry = last_close
    buf   = last_close * buffer_pct  # ~0.03% buffer beyond OB

    if direction == "Bullish":
        sl   = ob_low  - buf
        risk = entry - sl
        if risk <= 0: return None
        tp1  = entry + risk
        tp2  = find_next_liquidity(high_, low_, "Bullish", entry, SWING_LOOKBACK)
        rr   = round((tp2 - entry) / risk, 2)
    else:
        sl   = ob_high + buf
        risk = sl - entry
        if risk <= 0: return None
        tp1  = entry - risk
        tp2  = find_next_liquidity(high_, low_, "Bearish", entry, SWING_LOOKBACK)
        rr   = round((entry - tp2) / risk, 2)

    if rr < MIN_RR: return None

    dp = 5 if "JPY" not in symbol and symbol not in ["US30","BTCUSD"] else 3
    if symbol == "BTCUSD": dp = 2
    if symbol == "US30":   dp = 1

    return {
        "symbol":    symbol,
        "timeframe": tf_name,
        "type":      sig_type,
        "direction": direction,
        "entry":     round(entry, dp),
        "sl":        round(sl,    dp),
        "tp1":       round(tp1,   dp),
        "tp2":       round(tp2,   dp),
        "rr":        rr,
        "time":      datetime.now(pytz.timezone("Africa/Johannesburg"))
                             .strftime("%H:%M %Z"),
    }


# ── Detection functions ───────────────────────────────────────────────
def check_bos(df, symbol, tf_name):
    if len(df) < SWING_LOOKBACK + 5: return None
    h, l, o, c = df["High"], df["Low"], df["Open"], df["Close"]
    sh, sl, _, _ = swing_high_low(h.iloc[-(SWING_LOOKBACK+2):-1],
                                   l.iloc[-(SWING_LOOKBACK+2):-1],
                                   SWING_LOOKBACK)
    last = c.iloc[-1]
    if last > sh:
        ob_h, ob_l = find_ob(o, c, h, l, "Bullish", SWING_LOOKBACK)
        if ob_h: return build_signal(symbol, tf_name, "BOS", "Bullish",
                                     ob_h, ob_l, last, h, l)
    if last < sl:
        ob_h, ob_l = find_ob(o, c, h, l, "Bearish", SWING_LOOKBACK)
        if ob_h: return build_signal(symbol, tf_name, "BOS", "Bearish",
                                     ob_h, ob_l, last, h, l)
    return None


def check_choch(df, symbol, tf_name):
    if len(df) < SWING_LOOKBACK + 5: return None
    h, l, o, c = df["High"], df["Low"], df["Open"], df["Close"]

    downtrend = (h.iloc[-4] > h.iloc[-3] > h.iloc[-2] and
                 l.iloc[-4] > l.iloc[-3] > l.iloc[-2])
    uptrend   = (h.iloc[-4] < h.iloc[-3] < h.iloc[-2] and
                 l.iloc[-4] < l.iloc[-3] < l.iloc[-2])

    last = c.iloc[-1]
    if downtrend and last > h.iloc[-2]:
        ob_h, ob_l = find_ob(o, c, h, l, "Bullish", SWING_LOOKBACK)
        if ob_h: return build_signal(symbol, tf_name, "CHoCH", "Bullish",
                                     ob_h, ob_l, last, h, l)
    if uptrend and last < l.iloc[-2]:
        ob_h, ob_l = find_ob(o, c, h, l, "Bearish", SWING_LOOKBACK)
        if ob_h: return build_signal(symbol, tf_name, "CHoCH", "Bearish",
                                     ob_h, ob_l, last, h, l)
    return None


def check_fvg(df, symbol, tf_name):
    if len(df) < 5: return None
    h, l, o, c = df["High"], df["Low"], df["Open"], df["Close"]
    last = c.iloc[-1]

    bull_gap = l.iloc[-1] - h.iloc[-3]
    bear_gap = l.iloc[-3] - h.iloc[-1]
    threshold = last * 0.0002  # 0.02% minimum gap

    if bull_gap > threshold:
        return build_signal(symbol, tf_name, "FVG", "Bullish",
                            l.iloc[-1], h.iloc[-3], last, h, l)
    if bear_gap > threshold:
        return build_signal(symbol, tf_name, "FVG", "Bearish",
                            l.iloc[-3], h.iloc[-1], last, h, l)
    return None


def check_liq_sweep(df, symbol, tf_name):
    if len(df) < SWING_LOOKBACK + 5: return None
    h, l, o, c = df["High"], df["Low"], df["Open"], df["Close"]
    sh, sl, sh_i, sl_i = swing_high_low(h.iloc[-(SWING_LOOKBACK+2):-1],
                                         l.iloc[-(SWING_LOOKBACK+2):-1],
                                         SWING_LOOKBACK)
    last_high  = h.iloc[-1]
    last_low   = l.iloc[-1]
    last_close = c.iloc[-1]
    wick_min   = last_close * 0.0003

    if last_low < sl - wick_min and last_close > sl:
        ob_h = h.iloc[-(sl_i+2)]
        ob_l = l.iloc[-(sl_i+2)]
        return build_signal(symbol, tf_name, "Liq Sweep", "Bullish",
                            ob_h, ob_l, last_close, h, l)
    if last_high > sh + wick_min and last_close < sh:
        ob_h = h.iloc[-(sh_i+2)]
        ob_l = l.iloc[-(sh_i+2)]
        return build_signal(symbol, tf_name, "Liq Sweep", "Bearish",
                            ob_h, ob_l, last_close, h, l)
    return None


# ── Main scanner ─────────────────────────────────────────────────────
def run_scanner():
    signals = []
    for display_name, ticker in SYMBOLS.items():
        for tf_name, tf_cfg in TIMEFRAMES.items():
            df = get_data(ticker, tf_cfg["interval"], tf_cfg["period"])
            if df.empty or len(df) < 20:
                continue
            for check_fn in [check_choch, check_bos, check_fvg, check_liq_sweep]:
                sig = check_fn(df, display_name, tf_name)
                if sig:
                    signals.append(sig)
                    break  # one signal per symbol/tf combo
    # Sort: CHoCH first, then by R:R descending
    priority = {"CHoCH": 0, "BOS": 1, "Liq Sweep": 2, "FVG": 3}
    signals.sort(key=lambda x: (priority.get(x["type"], 9), -x["rr"]))
    return signals


# ── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/signals")
def api_signals():
    signals = run_scanner()
    return jsonify({
        "signals": signals,
        "count":   len(signals),
        "scanned": len(SYMBOLS) * len(TIMEFRAMES),
        "updated": datetime.now(pytz.timezone("Africa/Johannesburg"))
                           .strftime("%H:%M:%S %Z"),
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
