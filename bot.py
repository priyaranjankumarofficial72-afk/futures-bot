# =========================================================
# FUTURES SIGNAL BOT — Interactive Menu Version
# On-demand scanning with all 70 factors
# =========================================================

import asyncio
import csv
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
from aiohttp import web
from binance import AsyncClient
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# =========================================================
# CONFIG
# =========================================================
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET")
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")

MIN_VOLUME_USD = 10_000_000
ACCOUNT_BALANCE = 100.0
RISK_PER_TRADE = 0.01
MIN_RR = 2.0
LEVERAGE = 5
MIN_SCORE = 30
BATCH_SIZE = 10
BATCH_DELAY = 2
PORT = int(os.environ.get("PORT", 10000))

# =========================================================
# GLOBAL CLIENT
# =========================================================
binance_client = None

def body(c): return abs(c["close"] - c["open"])
def rng(c):  return c["high"] - c["low"]
def upper_wick(c): return c["high"] - max(c["open"], c["close"])
def lower_wick(c): return min(c["open"], c["close"]) - c["low"]
def is_bull(c): return c["close"] > c["open"]
def is_bear(c): return c["close"] < c["open"]
def mid(c): return (c["open"] + c["close"]) / 2
    # =========================================================
# DATA FETCH
# =========================================================
async def fetch_klines(client, symbol, interval, limit=250):
    try:
        klines = await client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        print(f"Klines error {symbol} {interval}: {e}")
        return None

async def fetch_oi(client, symbol):
    try:
        oi = await client.futures_open_interest(symbol=symbol)
        return float(oi["openInterest"])
    except Exception:
        return None

async def fetch_funding(client, symbol):
    try:
        fr = await client.futures_funding_rate(symbol=symbol, limit=1)
        return float(fr[0]["fundingRate"])
    except Exception:
        return None

async def fetch_ls_ratio(client, symbol):
    try:
        data = await client.futures_top_longshort_account_ratio(symbol=symbol, period="5m", limit=1)
        return float(data[-1]["longShortRatio"])
    except Exception:
        return None

async def fetch_taker(client, symbol):
    try:
        data = await client.futures_taker_volume(symbol=symbol, period="5m", limit=6)
        buy = sum(float(d["buyVol"]) for d in data)
        sell = sum(float(d["sellVol"]) for d in data)
        return {"buy": buy, "sell": sell, "ratio": buy / (sell or 1)}
    except Exception:
        return None

async def fetch_basis(client, symbol):
    try:
        mark = await client.futures_mark_price(symbol=symbol)
        mp = float(mark["markPrice"])
        ip = float(mark["indexPrice"])
        return {"mark": mp, "index": ip, "basis_pct": (mp - ip) / ip * 100}
    except Exception:
        return None

async def fetch_liquidations(client, symbol):
    try:
        data = await client.futures_futures_force_orders(symbol=symbol, limit=100)
        now = time.time() * 1000
        recent = [d for d in data if now - d["time"] < 300000]
        longs = sum(float(d["executedQty"]) * float(d["avgPrice"]) for d in recent if d["side"] == "SELL")
        shorts = sum(float(d["executedQty"]) * float(d["avgPrice"]) for d in recent if d["side"] == "BUY")
        return {"long_liq": longs, "short_liq": shorts}
    except Exception:
        return {"long_liq": 0, "short_liq": 0}

def fetch_news(symbol, hours=2):
    if not CRYPTOPANIC_API_KEY:
        return {"has_news": False, "count": 0}
    try:
        base = symbol.replace("USDT", "")
        r = requests.get("https://cryptopanic.com/api/v1/posts/", params={
            "auth_token": CRYPTOPANIC_API_KEY,
            "currencies": base,
            "public": "true",
            "kind": "news",
        }, timeout=10)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        items = [p for p in r.json().get("results", [])
                 if datetime.fromisoformat(p["published_at"].replace("Z","+00:00")) > cutoff]
        return {"has_news": len(items) > 0, "count": len(items)}
    except Exception:
        return {"has_news": False, "count": 0}
# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df):
    df["ema20"]  = EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"]  = EMAIndicator(df["close"], 50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()
    df["rsi"]    = RSIIndicator(df["close"], 14).rsi()
    macd = MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    df["vol_avg"] = df["volume"].rolling(20).mean()
    bb = BollingerBands(df["close"], 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)
    try:
        df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
    except Exception:
        df["adx"] = 20.0
    try:
        df["vwap"] = VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"],
            volume=df["volume"], window=14
        ).volume_weighted_average_price()
    except Exception:
        df["vwap"] = df["close"]
    return df

# =========================================================
# TREND & STRUCTURE
# =========================================================
def get_trend(df):
    if df is None or len(df) < 200:
        return "UNKNOWN"
    last = df.iloc[-1]
    if last["ema50"] > last["ema200"] and last["close"] > last["ema50"]:
        return "BULLISH"
    if last["ema50"] < last["ema200"] and last["close"] < last["ema50"]:
        return "BEARISH"
    return "RANGING"

def get_structure(df):
    if df is None or len(df) < 20:
        return "UNCLEAR"
    highs = df["high"].rolling(5, center=True).max()
    lows  = df["low"].rolling(5, center=True).min()
    sh = df["high"][df["high"] == highs].tail(3).tolist()
    sl = df["low"][df["low"] == lows].tail(3).tolist()
    if len(sh) >= 2 and len(sl) >= 2:
        hh = sh[-1] > sh[-2]
        hl = sl[-1] > sl[-2]
        lh = sh[-1] < sh[-2]
        ll = sl[-1] < sl[-2]
        if hh and hl: return "HH_HL"
        if lh and ll: return "LH_LL"
    return "UNCLEAR"

def detect_bos_choch(df, lookback=30):
    if df is None or len(df) < 15:
        return {"bos": None, "choch": None}
    data = df.tail(lookback)
    if len(data) < 10:
        return {"bos": None, "choch": None}
    swing_high = data["high"].iloc[-10:-1].max()
    swing_low  = data["low"].iloc[-10:-1].min()
    last_close = data["close"].iloc[-1]
    highs = data["high"].rolling(3, center=True).max()
    lows  = data["low"].rolling(3, center=True).min()
    ph = data["high"][data["high"] == highs].tail(2).tolist()
    pl = data["low"][data["low"] == lows].tail(2).tolist()
    bos, choch = None, None
    if len(ph) >= 2 and len(pl) >= 2:
        up   = ph[-1] > ph[-2] and pl[-1] > pl[-2]
        down = ph[-1] < ph[-2] and pl[-1] < pl[-2]
        if last_close > swing_high:
            bos = "BULLISH_BOS" if up else "BULLISH_CHOCH"
            if not up: choch = "BULLISH_CHOCH"
        elif last_close < swing_low:
            bos = "BEARISH_BOS" if down else "BEARISH_CHOCH"
            if not down: choch = "BEARISH_CHOCH"
    return {"bos": bos, "choch": choch}

def get_key_levels(df_daily):
    if df_daily is None or len(df_daily) < 2:
        return {"pdh": None, "pdl": None}
    prev = df_daily.iloc[-2]
    return {"pdh": prev["high"], "pdl": prev["low"]}

def near_key_level(price, levels, tol=0.003):
    for name, lvl in levels.items():
        if lvl is None:
            continue
        if abs(price - lvl) / lvl < tol:
            return name
    return None
    # =========================================================
# LIQUIDITY / ZONES
# =========================================================
def check_liquidity_sweep(df, levels):
    if df is None or len(df) < 2:
        return None
    last = df.iloc[-1]
    pdl = levels.get("pdl")
    pdh = levels.get("pdh")
    if pdl and last["low"] < pdl and last["close"] > pdl:
        return "BULLISH_SWEEP"
    if pdh and last["high"] > pdh and last["close"] < pdh:
        return "BEARISH_SWEEP"
    return None

def detect_supply_demand(df, lookback=100):
    if df is None or len(df) < 5:
        return {"demand": [], "supply": []}
    data = df.tail(lookback).reset_index(drop=True)
    demand, supply = [], []
    for i in range(2, len(data) - 1):
        c = data.iloc[i]
        n = data.iloc[i + 1]
        if (is_bear(c) and is_bull(n) and
            (n["close"] - n["open"]) > (c["open"] - c["close"]) * 1.5):
            demand.append({"low": c["low"], "high": c["high"]})
        if (is_bull(c) and is_bear(n) and
            (n["open"] - n["close"]) > (c["close"] - c["open"]) * 1.5):
            supply.append({"low": c["low"], "high": c["high"]})
    return {"demand": demand[-3:], "supply": supply[-3:]}

def price_in_zone(price, zones, tol=0.002):
    for z in zones:
        if z["low"] * (1 - tol) <= price <= z["high"] * (1 + tol):
            return True
    return False

def detect_fvg(df):
    if df is None or len(df) < 5:
        return []
    fvgs = []
    for i in range(2, min(len(df), 50)):
        c1 = df.iloc[i-2]
        c3 = df.iloc[i]
        if c3["low"] > c1["high"]:
            fvgs.append({"type": "bullish", "low": c1["high"], "high": c3["low"]})
        if c3["high"] < c1["low"]:
            fvgs.append({"type": "bearish", "low": c3["high"], "high": c1["low"]})
    return fvgs[-3:]

def premium_discount(df, lookback=100):
    if df is None or len(df) < 10:
        return "EQUILIBRIUM"
    high = df["high"].tail(lookback).max()
    low  = df["low"].tail(lookback).min()
    midp = (high + low) / 2
    price = df["close"].iloc[-1]
    if price > midp:  return "PREMIUM"
    if price < midp:  return "DISCOUNT"
    return "EQUILIBRIUM"

def volume_profile(df, bins=50, lookback=200):
    if df is None or len(df) < 20:
        return {"hvn": [], "lvn": [], "poc": None}
    data = df.tail(lookback)
    pmin, pmax = data["low"].min(), data["high"].max()
    if pmax == pmin:
        return {"hvn": [], "lvn": [], "poc": None}
    edges = np.linspace(pmin, pmax, bins + 1)
    vols = np.zeros(bins)
    for _, row in data.iterrows():
        li = max(0, np.searchsorted(edges, row["low"]) - 1)
        hi = min(bins - 1, np.searchsorted(edges, row["high"]) - 1)
        if hi >= li:
            v = row["volume"] / (hi - li + 1)
            for i in range(li, hi + 1):
                vols[i] += v
    poc_idx = int(np.argmax(vols))
    poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
    avg = vols.mean()
    hvn = [(edges[i] + edges[i+1]) / 2 for i in range(bins) if vols[i] > avg * 1.5]
    lvn = [(edges[i] + edges[i+1]) / 2 for i in range(bins) if vols[i] < avg * 0.5]
    return {"hvn": hvn, "lvn": lvn, "poc": poc}

def find_divergence(df, ind="rsi", lookback=40):
    if df is None or len(df) < 15:
        return None
    data = df.tail(lookback).reset_index(drop=True)
    if len(data) < 15:
        return None
    highs = data["high"].rolling(3, center=True).max()
    lows  = data["low"].rolling(3, center=True).min()
    hi_idx = data.index[data["high"] == highs].tolist()
    lo_idx = data.index[data["low"] == lows].tolist()
    if len(lo_idx) >= 2:
        p1, p2 = lo_idx[-2], lo_idx[-1]
        if data.loc[p2, "low"] < data.loc[p1, "low"] and data.loc[p2, ind] > data.loc[p1, ind]:
            return "BULLISH_DIVERGENCE"
    if len(hi_idx) >= 2:
        p1, p2 = hi_idx[-2], hi_idx[-1]
        if data.loc[p2, "high"] > data.loc[p1, "high"] and data.loc[p2, ind] < data.loc[p1, ind]:
            return "BEARISH_DIVERGENCE"
    return None
    # =========================================================
# CANDLESTICK PATTERNS (39 patterns)
# =========================================================
def is_doji(c): return body(c) <= rng(c) * 0.1
def is_dragonfly(c): return is_doji(c) and lower_wick(c) > rng(c) * 0.6 and upper_wick(c) < rng(c) * 0.1
def is_gravestone(c): return is_doji(c) and upper_wick(c) > rng(c) * 0.6 and lower_wick(c) < rng(c) * 0.1
def is_long_legged(c): return is_doji(c) and upper_wick(c) > rng(c) * 0.3 and lower_wick(c) > rng(c) * 0.3
def is_hammer(c): return lower_wick(c) > body(c) * 2 and upper_wick(c) < body(c) * 0.5 and body(c) > 0
def is_inv_hammer(c): return upper_wick(c) > body(c) * 2 and lower_wick(c) < body(c) * 0.5 and body(c) > 0
def is_shooting_star(c): return upper_wick(c) > body(c) * 2 and lower_wick(c) < body(c) * 0.5 and is_bear(c)
def is_marubozu_bull(c): return is_bull(c) and upper_wick(c) <= rng(c) * 0.05 and lower_wick(c) <= rng(c) * 0.05
def is_marubozu_bear(c): return is_bear(c) and upper_wick(c) <= rng(c) * 0.05 and lower_wick(c) <= rng(c) * 0.05
def is_spinning_top(c): return not is_doji(c) and body(c) < rng(c) * 0.3 and upper_wick(c) > body(c) and lower_wick(c) > body(c)

def is_bull_engulf(c1, c2):
    return is_bear(c1) and is_bull(c2) and c2["close"] > c1["open"] and c2["open"] < c1["close"]
def is_bear_engulf(c1, c2):
    return is_bull(c1) and is_bear(c2) and c2["close"] < c1["open"] and c2["open"] > c1["close"]
def is_piercing(c1, c2):
    return is_bear(c1) and is_bull(c2) and c2["open"] < c1["low"] and c2["close"] > mid(c1) and c2["close"] < c1["open"]
def is_dark_cloud(c1, c2):
    return is_bull(c1) and is_bear(c2) and c2["open"] > c1["high"] and c2["close"] < mid(c1) and c2["close"] > c1["open"]
def is_bull_harami(c1, c2):
    return is_bear(c1) and is_bull(c2) and c2["open"] > c1["close"] and c2["close"] < c1["open"]
def is_bear_harami(c1, c2):
    return is_bull(c1) and is_bear(c2) and c2["open"] < c1["close"] and c2["close"] > c1["open"]
def is_tweezer_bottom(c1, c2):
    return abs(c1["low"] - c2["low"]) / c1["low"] < 0.001 and is_bull(c2)
def is_tweezer_top(c1, c2):
    return abs(c1["high"] - c2["high"]) / c1["high"] < 0.001 and is_bear(c2)

def is_morning_star(c1, c2, c3):
    return is_bear(c1) and body(c2) < body(c1) * 0.5 and is_bull(c3) and c3["close"] > mid(c1)
def is_evening_star(c1, c2, c3):
    return is_bull(c1) and body(c2) < body(c1) * 0.5 and is_bear(c3) and c3["close"] < mid(c1)
def is_three_white(c1, c2, c3):
    return (is_bull(c1) and is_bull(c2) and is_bull(c3) and
            c2["close"] > c1["close"] and c3["close"] > c2["close"] and
            c2["open"] > c1["open"] and c2["open"] < c1["close"] and
            c3["open"] > c2["open"] and c3["open"] < c2["close"])
def is_three_black(c1, c2, c3):
    return (is_bear(c1) and is_bear(c2) and is_bear(c3) and
            c2["close"] < c1["close"] and c3["close"] < c2["close"] and
            c2["open"] < c1["open"] and c2["open"] > c1["close"] and
            c3["open"] < c2["open"] and c3["open"] > c2["close"])
def is_three_inside_up(c1, c2, c3):
    return is_bull_harami(c1, c2) and is_bull(c3) and c3["close"] > c1["open"]
def is_three_inside_down(c1, c2, c3):
    return is_bear_harami(c1, c2) and is_bear(c3) and c3["close"] < c1["open"]
def is_three_outside_up(c1, c2, c3):
    return is_bull_engulf(c1, c2) and is_bull(c3) and c3["close"] > c2["close"]
def is_three_outside_down(c1, c2, c3):
    return is_bear_engulf(c1, c2) and is_bear(c3) and c3["close"] < c2["close"]

def detect_patterns(df):
    if df is None or len(df) < 5:
        return []
    c1 = df.iloc[-5]; c2 = df.iloc[-4]; c3 = df.iloc[-3]; c4 = df.iloc[-2]; c5 = df.iloc[-1]
    p = []
    if is_doji(c5): p.append("DOJI")
    if is_dragonfly(c5): p.append("DRAGONFLY_DOJI")
    if is_gravestone(c5): p.append("GRAVESTONE_DOJI")
    if is_long_legged(c5): p.append("LONG_LEGGED_DOJI")
    if is_hammer(c5): p.append("HAMMER")
    if is_inv_hammer(c5): p.append("INVERTED_HAMMER")
    if is_shooting_star(c5): p.append("SHOOTING_STAR")
    if is_marubozu_bull(c5): p.append("BULLISH_MARUBOZU")
    if is_marubozu_bear(c5): p.append("BEARISH_MARUBOZU")
    if is_spinning_top(c5): p.append("SPINNING_TOP")
    if is_bull_engulf(c4, c5): p.append("BULLISH_ENGULFING")
    if is_bear_engulf(c4, c5): p.append("BEARISH_ENGULFING")
    if is_piercing(c4, c5): p.append("PIERCING_LINE")
    if is_dark_cloud(c4, c5): p.append("DARK_CLOUD_COVER")
    if is_bull_harami(c4, c5): p.append("BULLISH_HARAMI")
    if is_bear_harami(c4, c5): p.append("BEARISH_HARAMI")
    if is_tweezer_bottom(c4, c5): p.append("TWEEZER_BOTTOM")
    if is_tweezer_top(c4, c5): p.append("TWEEZER_TOP")
    if is_morning_star(c3, c4, c5): p.append("MORNING_STAR")
    if is_evening_star(c3, c4, c5): p.append("EVENING_STAR")
    if is_three_white(c3, c4, c5): p.append("THREE_WHITE_SOLDIERS")
    if is_three_black(c3, c4, c5): p.append("THREE_BLACK_CROWS")
    if is_three_inside_up(c3, c4, c5): p.append("THREE_INSIDE_UP")
    if is_three_inside_down(c3, c4, c5): p.append("THREE_INSIDE_DOWN")
    if is_three_outside_up(c3, c4, c5): p.append("THREE_OUTSIDE_UP")
    if is_three_outside_down(c3, c4, c5): p.append("THREE_OUTSIDE_DOWN")
    return p

BULLISH = {"HAMMER","INVERTED_HAMMER","BULLISH_ENGULFING","PIERCING_LINE","BULLISH_HARAMI",
           "TWEEZER_BOTTOM","MORNING_STAR","THREE_WHITE_SOLDIERS","THREE_INSIDE_UP",
           "THREE_OUTSIDE_UP","BULLISH_MARUBOZU","DRAGONFLY_DOJI"}
BEARISH = {"SHOOTING_STAR","BEARISH_ENGULFING","DARK_CLOUD_COVER","BEARISH_HARAMI",
           "TWEEZER_TOP","EVENING_STAR","THREE_BLACK_CROWS","THREE_INSIDE_DOWN",
           "THREE_OUTSIDE_DOWN","BEARISH_MARUBOZU","GRAVESTONE_DOJI"}
# =========================================================
# SCORING (70 FACTORS)
# =========================================================
def compute_score(s):
    score = 0
    if s["trend_4h"] in ("BULLISH","BEARISH"): score += 2
    if s["trend_1h"] in ("BULLISH","BEARISH"): score += 2
    if s["trend_1d"] in ("BULLISH","BEARISH"): score += 1
    if s["structure"] in ("HH_HL","LH_LL"): score += 2
    if s["trend_5m"] == s["trend_1h"]: score += 1
    if s["bos_choch"]["bos"]: score += 2
    if s["bos_choch"]["choch"]: score += 1
    if s["key_level"]: score += 2
    if s["sweep"]: score += 3
    if s["zones"]["demand"] or s["zones"]["supply"]: score += 2
    if s["fvg"]: score += 1
    if s["pd_zone"] in ("PREMIUM","DISCOUNT"): score += 1
    if s["volume"]: score += 3
    if s["near_hvn"]: score += 1
    if s["near_lvn"]: score += 1
    if s["oi"]: score += 1
    if s["funding"] is not None: score += 1
    if s["ls_ratio"]: score += 1
    if s["taker"]: score += 1
    if s["basis"]: score += 1
    if s["liq"] and (s["liq"]["long_liq"] > 0 or s["liq"]["short_liq"] > 0): score += 1
    if s["vwap_ok"]: score += 1
    if s["rsi_ok"]: score += 1
    if s["macd_ok"]: score += 1
    if s["adx_ok"]: score += 1
    if s["divergence"]: score += 2
    if s["atr"]: score += 1
    if s["bb_squeeze"]: score += 1
    if s["patterns_bull"]: score += 3
    if s["patterns_bear"]: score += 3
    if len(s["patterns_bull"] + s["patterns_bear"]) >= 2: score += 2
    if s["news_ok"]: score += 2
    if s["session"] in ("LONDON","NY"): score += 1
    if s["kill_zone"]: score += 1
    if s["rr"] >= MIN_RR: score += 3
    if s["sl_dist"] > 0: score += 1
    return score

def get_session():
    h = datetime.now(timezone.utc).hour
    if 0 <= h < 8: return "ASIAN"
    if 8 <= h < 16: return "LONDON"
    if 13 <= h < 21: return "NY"
    return "OFF"

def is_kill_zone():
    return datetime.now(timezone.utc).hour in (7, 8, 13, 14, 15)

# =========================================================
# MASTER ANALYZER
# =========================================================
async def analyze_symbol(client, symbol):
    try:
        df_4h  = add_indicators(await fetch_klines(client, symbol, "4h"))
        df_1h  = add_indicators(await fetch_klines(client, symbol, "1h"))
        df_15m = add_indicators(await fetch_klines(client, symbol, "15m"))
        df_5m  = add_indicators(await fetch_klines(client, symbol, "5m"))
        df_1d  = await fetch_klines(client, symbol, "1d", 5)
        if any(x is None for x in [df_4h, df_1h, df_15m, df_5m, df_1d]):
            return None
    except Exception as e:
        print(f"analyze_symbol error {symbol}: {e}")
        return None

    trend_4h = get_trend(df_4h)
    trend_1h = get_trend(df_1h)
    trend_5m = get_trend(df_5m)
    trend_1d = get_trend(df_1d)
    structure = get_structure(df_1h)
    bos_choch = detect_bos_choch(df_15m)

    levels = get_key_levels(df_1d)
    price = df_15m.iloc[-1]["close"]
    key_level = near_key_level(price, levels)
    sweep = check_liquidity_sweep(df_15m, levels)
    zones = detect_supply_demand(df_1h)
    zones_hit = {
        "demand": price_in_zone(price, zones["demand"]),
        "supply": price_in_zone(price, zones["supply"]),
    }
    fvgs = detect_fvg(df_15m)
    fvg_ok = any(f["type"] == "bullish" and f["low"] <= price <= f["high"] for f in fvgs) or \
             any(f["type"] == "bearish" and f["low"] <= price <= f["high"] for f in fvgs)
    pd_zone = premium_discount(df_15m)

    vol_ok = df_15m.iloc[-1]["volume"] > df_15m.iloc[-1]["vol_avg"] * 1.5
    vp = volume_profile(df_15m)
    near_hvn = any(abs(price - n) / price < 0.005 for n in vp["hvn"])
    near_lvn = any(abs(price - n) / price < 0.005 for n in vp["lvn"])

    oi = await fetch_oi(client, symbol)
    funding = await fetch_funding(client, symbol)
    ls_ratio = await fetch_ls_ratio(client, symbol)
    taker = await fetch_taker(client, symbol)
    basis = await fetch_basis(client, symbol)
    liq = await fetch_liquidations(client, symbol)
    news = fetch_news(symbol)

    rsi = df_15m.iloc[-1]["rsi"]
    macd_now = df_15m.iloc[-1]["macd"]
    macd_sig = df_15m.iloc[-1]["macd_signal"]
    adx = df_15m.iloc[-1]["adx"]
    atr = df_15m.iloc[-1]["atr"]
    vwap = df_15m.iloc[-1]["vwap"]

    patterns_15m = detect_patterns(df_15m)
    patterns_5m = detect_patterns(df_5m)
    bull_p = [p for p in patterns_15m + patterns_5m if p in BULLISH]
    bear_p = [p for p in patterns_15m + patterns_5m if p in BEARISH]

    div = find_divergence(df_15m, "rsi")

    base = {
        "symbol": symbol, "price": price,
        "trend_4h": trend_4h, "trend_1h": trend_1h,
        "trend_5m": trend_5m, "trend_1d": trend_1d,
        "structure": structure, "bos_choch": bos_choch,
        "key_level": key_level, "sweep": sweep,
        "zones": zones_hit, "fvg": fvg_ok, "pd_zone": pd_zone,
        "volume": vol_ok, "near_hvn": near_hvn, "near_lvn": near_lvn,
        "oi": oi, "funding": funding, "ls_ratio": ls_ratio,
        "taker": taker, "basis": basis, "liq": liq,
        "news_ok": not news["has_news"],
        "rsi": round(rsi, 2), "adx": round(adx, 2),
        "atr": round(atr, 4), "vwap": round(vwap, 4),
        "patterns_bull": bull_p, "patterns_bear": bear_p,
        "divergence": div, "session": get_session(),
        "kill_zone": is_kill_zone(),
    }

    long_ok = (
        trend_4h == "BULLISH" and trend_1h == "BULLISH" and
        structure == "HH_HL" and
        sweep == "BULLISH_SWEEP" and
        (zones_hit["demand"] or fvg_ok or pd_zone == "DISCOUNT") and
        bull_p and vol_ok and rsi < 65 and macd_now > macd_sig and
        adx > 20 and base["news_ok"] and price > vwap
    )

    short_ok = (
        trend_4h == "BEARISH" and trend_1h == "BEARISH" and
        structure == "LH_LL" and
        sweep == "BEARISH_SWEEP" and
        (zones_hit["supply"] or fvg_ok or pd_zone == "PREMIUM") and
        bear_p and vol_ok and rsi > 35 and macd_now < macd_sig and
        adx > 20 and base["news_ok"] and price < vwap
    )

    if long_ok:
        entry = price
        sl = min(df_15m["low"].tail(10)) - atr * 0.5
        risk = entry - sl
        base.update({
            "side": "LONG", "entry": entry, "sl": sl,
            "tp1": entry + risk, "tp2": entry + risk * 2, "tp3": entry + risk * 3,
            "sl_dist": risk, "rr": (entry + risk * 2 - entry) / risk if risk > 0 else 0,
            "rsi_ok": rsi < 65, "macd_ok": True, "adx_ok": adx > 20,
            "vwap_ok": True, "bb_squeeze": df_15m.iloc[-1]["bb_squeeze"],
        })
        base["score"] = compute_score(base)
        return base

    if short_ok:
        entry = price
        sl = max(df_15m["high"].tail(10)) + atr * 0.5
        risk = sl - entry
        base.update({
            "side": "SHORT", "entry": entry, "sl": sl,
            "tp1": entry - risk, "tp2": entry - risk * 2, "tp3": entry - risk * 3,
            "sl_dist": risk, "rr": (entry - (entry - risk * 2)) / risk if risk > 0 else 0,
            "rsi_ok": rsi > 35, "macd_ok": True, "adx_ok": adx > 20,
            "vwap_ok": True, "bb_squeeze": df_15m.iloc[-1]["bb_squeeze"],
        })
        base["score"] = compute_score(base)
        return base

    return None
    # =========================================================
# SCANNER FUNCTIONS
# =========================================================
async def get_top10(client):
    tickers = await client.futures_ticker()
    perps = [t for t in tickers if t["symbol"].endswith("USDT")
             and not any(x in t["symbol"] for x in ["_","UP","DOWN","BULL","BEAR"])]
    perps.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in perps[:10]]

async def get_gainers(client, n=20):
    tickers = await client.futures_ticker()
    clean = []
    for t in tickers:
        s = t["symbol"]
        if not s.endswith("USDT"): continue
        if any(x in s for x in ["_","UP","DOWN","BULL","BEAR"]): continue
        try:
            vol = float(t["quoteVolume"])
            pct = float(t["priceChangePercent"])
            if vol < MIN_VOLUME_USD: continue
            clean.append({"symbol": s, "pct": pct})
        except: continue
    clean.sort(key=lambda x: x["pct"], reverse=True)
    return [c["symbol"] for c in clean[:n]]

async def get_losers(client, n=20):
    tickers = await client.futures_ticker()
    clean = []
    for t in tickers:
        s = t["symbol"]
        if not s.endswith("USDT"): continue
        if any(x in s for x in ["_","UP","DOWN","BULL","BEAR"]): continue
        try:
            vol = float(t["quoteVolume"])
            pct = float(t["priceChangePercent"])
            if vol < MIN_VOLUME_USD: continue
            clean.append({"symbol": s, "pct": pct})
        except: continue
    clean.sort(key=lambda x: x["pct"])
    return [c["symbol"] for c in clean[:n]]

async def get_all_coins(client):
    top = await get_top10(client)
    gainers = await get_gainers(client, 20)
    losers = await get_losers(client, 20)
    seen = set(); final = []
    for s in top + gainers + losers:
        if s not in seen:
            seen.add(s); final.append(s)
    return final

# =========================================================
# SCAN WITH PROGRESS
# =========================================================
async def run_scan(client, symbols, update_msg, label="Scanning"):
    start = time.time()
    best = None
    total = len(symbols)

    for i, sym in enumerate(symbols):
        elapsed = int(time.time() - start)
        if i > 0:
            remaining = int((elapsed / i) * (total - i))
        else:
            remaining = 0

        if i % 2 == 0:
            pct = i * 10 // max(total, 1)
            bar = "█" * pct + "░" * (10 - pct)
            best_txt = f"Best: {best['score']}/70 ({best['symbol']})" if best else "Best: searching..."
            text = (
                f"🎯 <b>{label}</b>\n\n"
                f"Progress: [{bar}] {i}/{total}\n"
                f"Current: {sym}\n"
                f"Elapsed: {elapsed}s\n"
                f"Est. left: ~{remaining}s\n"
                f"{best_txt}"
            )
            try:
                await update_msg.edit_text(text, parse_mode="HTML")
            except Exception:
                pass

        result = await analyze_symbol(client, sym)
        if result and result.get("score", 0) > (best["score"] if best else 0):
            best = result

        await asyncio.sleep(0.2)

    return best, int(time.time() - start)

# =========================================================
# MESSAGE FORMAT
# =========================================================
def fmt_signal(s):
    emoji = "🟢" if s["side"] == "LONG" else "🔴"
    return f"""
{emoji} <b>{s['symbol']} — {s['side']}</b>
🎯 Score: <b>{s['score']}/70</b>

📊 <b>CONTEXT</b>
• 4H Trend: {s['trend_4h']}
• 1H Trend: {s['trend_1h']}
• Structure: {s['structure']}
• BOS: {s['bos_choch']['bos'] or '-'}
• Sweep: {s['sweep'] or '-'}
• Zone: {s['pd_zone']}

🕯️ <b>PATTERNS</b>
• Bull: {', '.join(s['patterns_bull']) or '-'}
• Bear: {', '.join(s['patterns_bear']) or '-'}
• Divergence: {s['divergence'] or '-'}

📈 <b>INDICATORS</b>
• RSI: {s['rsi']} | ADX: {s['adx']}
• ATR: {s['atr']} | VWAP: {s['vwap']}
• Volume: {'✅' if s['volume'] else '❌'}

💰 <b>TRADE PLAN</b>
• Entry: {round(s['entry'],4)}
• Stop-Loss: {round(s['sl'],4)}
• TP1: {round(s['tp1'],4)} (50%, SL→BE)
• TP2: {round(s['tp2'],4)} (30%, trail)
• TP3: {round(s['tp3'],4)} (20%)

⚖️ R:R = 1:{round(s['rr'],2)}
⏰ {datetime.now(timezone.utc).strftime('%H:%M')} UTC
"""

def fmt_no_signal(label, scanned, duration, best):
    best_txt = f"Best score: {best['score']}/70 ({best['symbol']})" if best else "No score info"
    return (
        f"🎯 <b>{label} Complete</b>\n\n"
        f"Scanned: {scanned} coins\n"
        f"Duration: {duration}s\n"
        f"{best_txt}\n\n"
        f"❌ No valid setup (below min score {MIN_SCORE})"
    )

# =========================================================
# TELEGRAM MENU
# =========================================================
def main_menu():
    kb = [
        [InlineKeyboardButton("⚡ Quick Signal (5)", callback_data="quick"),
         InlineKeyboardButton("🎯 Deep Analysis (50)", callback_data="deep")],
        [InlineKeyboardButton("📊 Top 10 Coins", callback_data="top10"),
         InlineKeyboardButton("📈 Top 20 Gainers", callback_data="gainers")],
        [InlineKeyboardButton("📉 Top 20 Losers", callback_data="losers"),
         InlineKeyboardButton("🔍 Scan Single Coin", callback_data="single")],
        [InlineKeyboardButton("📋 Help", callback_data="help")],
    ]
    return InlineKeyboardMarkup(kb)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Futures Signal Bot</b>\n\n"
        "Choose what to scan:",
        reply_markup=main_menu(),
        parse_mode="HTML"
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    msg = q.message

    global binance_client
    if binance_client is None:
        binance_client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)

    try:
        if data == "quick":
            await q.edit_message_text("⚡ Quick scan starting...")
            syms = (await get_top10(binance_client))[:5]
            best, dur = await run_scan(binance_client, syms, msg, "Quick Scan (5 coins)")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML",
                                          reply_markup=main_menu())
            else:
                await q.edit_message_text(
                    fmt_no_signal("Quick Scan", len(syms), dur, best),
                    parse_mode="HTML", reply_markup=main_menu())

        elif data == "deep":
            await q.edit_message_text("🎯 Deep analysis starting (may take 3-5 min)...")
            syms = await get_all_coins(binance_client)
            best, dur = await run_scan(binance_client, syms, msg, "Deep Analysis")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML",
                                          reply_markup=main_menu())
            else:
                await q.edit_message_text(
                    fmt_no_signal("Deep Analysis", len(syms), dur, best),
                    parse_mode="HTML", reply_markup=main_menu())

        elif data == "top10":
            await q.edit_message_text("📊 Scanning Top 10...")
            syms = await get_top10(binance_client)
            best, dur = await run_scan(binance_client, syms, msg, "Top 10 Scan")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML",
                                          reply_markup=main_menu())
            else:
                await q.edit_message_text(
                    fmt_no_signal("Top 10 Scan", len(syms), dur, best),
                    parse_mode="HTML", reply_markup=main_menu())

        elif data == "gainers":
            await q.edit_message_text("📈 Scanning Top 20 Gainers...")
            syms = await get_gainers(binance_client, 20)
            best, dur = await run_scan(binance_client, syms, msg, "Top Gainers Scan")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML",
                                          reply_markup=main_menu())
            else:
                await q.edit_message_text(
                    fmt_no_signal("Gainers Scan", len(syms), dur, best),
                    parse_mode="HTML", reply_markup=main_menu())

        elif data == "losers":
            await q.edit_message_text("📉 Scanning Top 20 Losers...")
            syms = await get_losers(binance_client, 20)
            best, dur = await run_scan(binance_client, syms, msg, "Top Losers Scan")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML",
                                          reply_markup=main_menu())
            else:
                await q.edit_message_text(
                    fmt_no_signal("Losers Scan", len(syms), dur, best),
                    parse_mode="HTML", reply_markup=main_menu())

        elif data == "single":
            await q.edit_message_text(
                "🔍 Send me a symbol:\n\nExample: <code>BTCUSDT</code>",
                parse_mode="HTML"
            )

        elif data == "help":
            await q.edit_message_text(
                "📖 <b>How to use</b>\n\n"
                "• ⚡ Quick: Top 5 coins, fast\n"
                "• 🎯 Deep: All ~50 coins, full analysis\n"
                "• 📊 Top 10: By volume\n"
                "• 📈 Gainers / 📉 Losers: By 24h %\n"
                "• 🔍 Single: Type any symbol\n\n"
                f"Min score to signal: {MIN_SCORE}/70",
                parse_mode="HTML", reply_markup=main_menu())

    except Exception as e:
        print(f"Button error: {e}")
        try:
            await q.edit_message_text(
                f"⚠️ Error: {str(e)[:100]}",
                reply_markup=main_menu()
            )
        except Exception:
            pass

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global binance_client
    sym = update.message.text.strip().upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"

    if binance_client is None:
        binance_client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)

    msg = await update.message.reply_text(f"🔍 Scanning {sym}...")
    try:
        result = await analyze_symbol(binance_client, sym)
        if result and result["score"] >= MIN_SCORE:
            await msg.edit_text(fmt_signal(result), parse_mode="HTML",
                                reply_markup=main_menu())
        else:
            score = result["score"] if result else 0
            await msg.edit_text(
                f"🔍 <b>{sym}</b>\n\nScore: {score}/70\n❌ Below min score {MIN_SCORE}",
                parse_mode="HTML", reply_markup=main_menu())
    except Exception as e:
        await msg.edit_text(f"⚠️ Error scanning {sym}: {str(e)[:100]}",
                            reply_markup=main_menu())
        # =========================================================
# ANTI-SLEEP WEB SERVER
# =========================================================
async def health(request):
    return web.Response(text="OK")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"✅ Health server running on port {PORT}")

# =========================================================
# MAIN RUNNER
# =========================================================
async def main():
    global binance_client
    binance_client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    await start_web_server()

    print("🤖 Bot started — interactive mode")
    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text="🤖 <b>Bot is LIVE</b>\n\nSend /start to open the menu.",
            parse_mode="HTML"
        )
    except Exception as e:
        print(f"Startup message error: {e}")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    # Keep running forever
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
