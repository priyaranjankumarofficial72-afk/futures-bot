# =========================================================
# FUTURES SIGNAL BOT — Bybit Data (30 Factors + 15 Patterns)
# Interactive Menu + Auto Scan every 30 min
# =========================================================

import asyncio
import csv
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import aiohttp
import requests
from aiohttp import web
from pybit.unified_trading import HTTP
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# =========================================================
# CONFIG
# =========================================================
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MIN_VOLUME_USD = 10_000_000
MIN_SCORE = 20
PORT = int(os.environ.get("PORT", 10000))
AUTO_SCAN_MINUTES = 30

bybit = HTTP(testnet=False)

# =========================================================
# HELPERS
# =========================================================
def body(c): return abs(c["close"] - c["open"])
def rng(c):  return c["high"] - c["low"]
def upper_wick(c): return c["high"] - max(c["open"], c["close"])
def lower_wick(c): return min(c["open"], c["close"]) - c["low"]
def is_bull(c): return c["close"] > c["open"]
def is_bear(c): return c["close"] < c["open"]
def mid(c): return (c["open"] + c["close"]) / 2

# =========================================================
# BYBIT DATA FETCH
# =========================================================
INTERVAL_MAP = {
    "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "4h": "240", "1d": "D",
}

async def fetch_klines(symbol, interval, limit=250):
    try:
        loop = asyncio.get_event_loop()
        iv = INTERVAL_MAP.get(interval, interval)
        data = await loop.run_in_executor(
            None,
            lambda: bybit.get_kline(
                category="linear", symbol=symbol,
                interval=iv, limit=min(limit, 1000)
            )
        )
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=[
            "time","open","high","low","close","volume","turnover"
        ])
        for c in ["open","high","low","close","volume","turnover"]:
            df[c] = df[c].astype(float)
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Klines error {symbol} {interval}: {e}")
        return None

async def fetch_ticker():
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: bybit.get_tickers(category="linear")
        )
        return data.get("result", {}).get("list", [])
    except Exception as e:
        print(f"Ticker error: {e}")
        return []

# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df):
    try:
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
    except Exception as e:
        print(f"Indicator error: {e}")
    return df

# =========================================================
# TREND, STRUCTURE, LEVELS
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
        hh = sh[-1] > sh[-2]; hl = sl[-1] > sl[-2]
        lh = sh[-1] < sh[-2]; ll = sl[-1] < sl[-2]
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
        if lvl is None: continue
        if abs(price - lvl) / lvl < tol:
            return name
    return None

def check_liquidity_sweep(df, levels):
    if df is None or len(df) < 2:
        return None
    last = df.iloc[-1]
    pdl = levels.get("pdl"); pdh = levels.get("pdh")
    if pdl and last["low"] < pdl and last["close"] > pdl:
        return "BULLISH_SWEEP"
    if pdh and last["high"] > pdh and last["close"] < pdh:
        return "BEARISH_SWEEP"
    return None
    # =========================================================
# CANDLESTICK PATTERNS (15)
# =========================================================
def is_doji(c): return body(c) <= rng(c) * 0.1
def is_hammer(c): return lower_wick(c) > body(c) * 2 and upper_wick(c) < body(c) * 0.5 and body(c) > 0
def is_inv_hammer(c): return upper_wick(c) > body(c) * 2 and lower_wick(c) < body(c) * 0.5 and body(c) > 0
def is_shooting_star(c): return upper_wick(c) > body(c) * 2 and lower_wick(c) < body(c) * 0.5 and is_bear(c)
def is_marubozu_bull(c): return is_bull(c) and upper_wick(c) <= rng(c) * 0.05 and lower_wick(c) <= rng(c) * 0.05
def is_marubozu_bear(c): return is_bear(c) and upper_wick(c) <= rng(c) * 0.05 and lower_wick(c) <= rng(c) * 0.05

def is_bull_engulf(c1, c2):
    return is_bear(c1) and is_bull(c2) and c2["close"] > c1["open"] and c2["open"] < c1["close"]
def is_bear_engulf(c1, c2):
    return is_bull(c1) and is_bear(c2) and c2["close"] < c1["open"] and c2["open"] > c1["close"]
def is_piercing(c1, c2):
    return is_bear(c1) and is_bull(c2) and c2["open"] < c1["low"] and c2["close"] > mid(c1) and c2["close"] < c1["open"]
def is_dark_cloud(c1, c2):
    return is_bull(c1) and is_bear(c2) and c2["open"] > c1["high"] and c2["close"] < mid(c1) and c2["close"] > c1["open"]
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
            c2["close"] > c1["close"] and c3["close"] > c2["close"])
def is_three_black(c1, c2, c3):
    return (is_bear(c1) and is_bear(c2) and is_bear(c3) and
            c2["close"] < c1["close"] and c3["close"] < c2["close"])

def detect_patterns(df):
    if df is None or len(df) < 5:
        return []
    c1 = df.iloc[-5]; c2 = df.iloc[-4]; c3 = df.iloc[-3]; c4 = df.iloc[-2]; c5 = df.iloc[-1]
    p = []
    if is_doji(c5): p.append("DOJI")
    if is_hammer(c5): p.append("HAMMER")
    if is_inv_hammer(c5): p.append("INVERTED_HAMMER")
    if is_shooting_star(c5): p.append("SHOOTING_STAR")
    if is_marubozu_bull(c5): p.append("BULLISH_MARUBOZU")
    if is_marubozu_bear(c5): p.append("BEARISH_MARUBOZU")
    if is_bull_engulf(c4, c5): p.append("BULLISH_ENGULFING")
    if is_bear_engulf(c4, c5): p.append("BEARISH_ENGULFING")
    if is_piercing(c4, c5): p.append("PIERCING_LINE")
    if is_dark_cloud(c4, c5): p.append("DARK_CLOUD_COVER")
    if is_tweezer_bottom(c4, c5): p.append("TWEEZER_BOTTOM")
    if is_tweezer_top(c4, c5): p.append("TWEEZER_TOP")
    if is_morning_star(c3, c4, c5): p.append("MORNING_STAR")
    if is_evening_star(c3, c4, c5): p.append("EVENING_STAR")
    if is_three_white(c3, c4, c5): p.append("THREE_WHITE_SOLDIERS")
    if is_three_black(c3, c4, c5): p.append("THREE_BLACK_CROWS")
    return p

BULLISH_P = {"HAMMER","INVERTED_HAMMER","BULLISH_ENGULFING","PIERCING_LINE",
             "TWEEZER_BOTTOM","MORNING_STAR","THREE_WHITE_SOLDIERS","BULLISH_MARUBOZU"}
BEARISH_P = {"SHOOTING_STAR","BEARISH_ENGULFING","DARK_CLOUD_COVER",
             "TWEEZER_TOP","EVENING_STAR","THREE_BLACK_CROWS","BEARISH_MARUBOZU"}

# =========================================================
# SCORING (30 FACTORS)
# =========================================================
def compute_score(s):
    score = 0
    if s["trend_4h"] in ("BULLISH","BEARISH"): score += 2
    if s["trend_1h"] in ("BULLISH","BEARISH"): score += 2
    if s["trend_1d"] in ("BULLISH","BEARISH"): score += 1
    if s["trend_5m"] == s["trend_1h"]: score += 1
    if s["structure"] in ("HH_HL","LH_LL"): score += 2
    if s["bos_choch"]["bos"]: score += 2
    if s["bos_choch"]["choch"]: score += 1
    if s["key_level"]: score += 2
    if s["sweep"]: score += 3
    if s["pdh_pdl_near"]: score += 2
    if s["volume"]: score += 2
    if s["ema20_ok"]: score += 1
    if s["ema50_ok"]: score += 1
    if s["ema200_ok"]: score += 1
    if s["rsi_ok"]: score += 1
    if s["macd_ok"]: score += 1
    if s["atr_ok"]: score += 1
    if s["bb_ok"]: score += 1
    if s["bb_squeeze"]: score += 1
    if s["adx_ok"]: score += 1
    if s["patterns_bull"]: score += 3
    if s["patterns_bear"]: score += 3
    if len(s["patterns_bull"] + s["patterns_bear"]) >= 2: score += 1
    if s["vwap_ok"]: score += 1
    if s["rr"] >= 2.0: score += 2
    if s["sl_dist"] > 0: score += 1
    return score

def stars(score):
    if score >= 25: return "⭐⭐⭐⭐⭐"
    if score >= 22: return "⭐⭐⭐⭐"
    if score >= 20: return "⭐⭐⭐"
    if score >= 18: return "⭐⭐"
    if score >= 15: return "⭐"
    return ""

# =========================================================
# MASTER ANALYZER
# =========================================================
async def analyze_symbol(symbol):
    try:
        df_4h  = add_indicators(await fetch_klines(symbol, "4h"))
        df_1h  = add_indicators(await fetch_klines(symbol, "1h"))
        df_15m = add_indicators(await fetch_klines(symbol, "15m"))
        df_5m  = add_indicators(await fetch_klines(symbol, "5m"))
        df_1d  = await fetch_klines(symbol, "1d", 5)
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
    pdh_pdl_near = key_level is not None

    vol_ok = df_15m.iloc[-1]["volume"] > df_15m.iloc[-1]["vol_avg"] * 1.5
    rsi = df_15m.iloc[-1]["rsi"]
    macd_now = df_15m.iloc[-1]["macd"]
    macd_sig = df_15m.iloc[-1]["macd_signal"]
    atr = df_15m.iloc[-1]["atr"]
    vwap = df_15m.iloc[-1]["vwap"]
    adx = df_15m.iloc[-1]["adx"]
    ema20 = df_15m.iloc[-1]["ema20"]
    ema50 = df_15m.iloc[-1]["ema50"]
    ema200 = df_15m.iloc[-1]["ema200"]
    bb_squeeze = bool(df_15m.iloc[-1]["bb_squeeze"])

    patterns = detect_patterns(df_15m)
    bull_p = [p for p in patterns if p in BULLISH_P]
    bear_p = [p for p in patterns if p in BEARISH_P]

    base = {
        "symbol": symbol, "price": price,
        "trend_4h": trend_4h, "trend_1h": trend_1h,
        "trend_5m": trend_5m, "trend_1d": trend_1d,
        "structure": structure, "bos_choch": bos_choch,
        "key_level": key_level, "sweep": sweep, "pdh_pdl_near": pdh_pdl_near,
        "volume": vol_ok, "rsi": round(rsi, 2),
        "atr": round(atr, 4), "vwap": round(vwap, 4),
        "adx": round(adx, 2), "bb_squeeze": bb_squeeze,
        "patterns_bull": bull_p, "patterns_bear": bear_p,
        "ema20_ok": price > ema20, "ema50_ok": price > ema50,
        "ema200_ok": price > ema200,
        "rsi_ok": rsi < 65 or rsi > 35,
        "macd_ok": macd_now > macd_sig or macd_now < macd_sig,
        "atr_ok": atr > 0,
        "bb_ok": True,
        "adx_ok": adx > 20,
    }

    long_ok = (
        trend_4h == "BULLISH" and trend_1h == "BULLISH" and
        structure == "HH_HL" and
        sweep == "BULLISH_SWEEP" and
        bull_p and vol_ok and rsi < 65 and macd_now > macd_sig and
        price > vwap
    )

    short_ok = (
        trend_4h == "BEARISH" and trend_1h == "BEARISH" and
        structure == "LH_LL" and
        sweep == "BEARISH_SWEEP" and
        bear_p and vol_ok and rsi > 35 and macd_now < macd_sig and
        price < vwap
    )

    if long_ok:
        entry = price
        sl = min(df_15m["low"].tail(10)) - atr * 0.5
        risk = entry - sl
        base.update({
            "side": "LONG", "entry": entry, "sl": sl,
            "tp1": entry + risk, "tp2": entry + risk * 2, "tp3": entry + risk * 3,
            "sl_dist": risk, "rr": 2.0,
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
            "sl_dist": risk, "rr": 2.0,
        })
        base["score"] = compute_score(base)
        return base

    return None
    # =========================================================
# SCANNER
# =========================================================
async def get_top10():
    tickers = await fetch_ticker()
    if not tickers:
        return []
    perps = [t for t in tickers if t["symbol"].endswith("USDT")]
    perps.sort(key=lambda x: float(x.get("turnover24h", 0)), reverse=True)
    return [t["symbol"] for t in perps[:10]]

async def get_gainers(n=20):
    tickers = await fetch_ticker()
    if not tickers:
        return []
    clean = []
    for t in tickers:
        s = t["symbol"]
        if not s.endswith("USDT"): continue
        try:
            if float(t.get("turnover24h", 0)) < MIN_VOLUME_USD: continue
            clean.append({"symbol": s, "pct": float(t.get("price24hPcnt", 0)) * 100})
        except: continue
    clean.sort(key=lambda x: x["pct"], reverse=True)
    return [c["symbol"] for c in clean[:n]]

async def get_losers(n=20):
    tickers = await fetch_ticker()
    if not tickers:
        return []
    clean = []
    for t in tickers:
        s = t["symbol"]
        if not s.endswith("USDT"): continue
        try:
            if float(t.get("turnover24h", 0)) < MIN_VOLUME_USD: continue
            clean.append({"symbol": s, "pct": float(t.get("price24hPcnt", 0)) * 100})
        except: continue
    clean.sort(key=lambda x: x["pct"])
    return [c["symbol"] for c in clean[:n]]

async def get_all_coins():
    top = await get_top10()
    gain = await get_gainers(20)
    lose = await get_losers(20)
    seen = set(); final = []
    for s in top + gain + lose:
        if s not in seen:
            seen.add(s); final.append(s)
    return final

# =========================================================
# SCAN WITH PROGRESS
# =========================================================
async def run_scan(symbols, update_msg, label, top_n=5):
    start = time.time()
    signals = []
    total = len(symbols)

    for i, sym in enumerate(symbols):
        elapsed = int(time.time() - start)
        remaining = int((elapsed / i) * (total - i)) if i > 0 else 0

        if i % 2 == 0 and update_msg is not None:
            pct = i * 10 // max(total, 1)
            bar = "█" * pct + "░" * (10 - pct)
            found_txt = f"Found: {len(signals)} signals" if signals else "Found: scanning..."
            txt = (
                f"🎯 <b>{label}</b>\n\n"
                f"Progress: [{bar}] {i}/{total}\n"
                f"Current: {sym}\n"
                f"Elapsed: {elapsed}s | Left: ~{remaining}s\n"
                f"{found_txt}"
            )
            try:
                await update_msg.edit_text(txt, parse_mode="HTML")
            except Exception:
                pass

        result = await analyze_symbol(sym)
        if result and result.get("score", 0) >= MIN_SCORE:
            signals.append(result)
        await asyncio.sleep(0.3)

    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals[:top_n], int(time.time() - start)

# =========================================================
# MESSAGE FORMAT
# =========================================================
def fmt_signal_block(s, idx):
    emoji = "🟢" if s["side"] == "LONG" else "🔴"
    star = stars(s["score"])
    return f"""
{emoji} <b>#{idx} {s['symbol']} — {s['side']}</b>  {star}
🎯 Score: <b>{s['score']}/30</b>

💰 Entry: {round(s['entry'],4)}
🛑 SL: {round(s['sl'],4)}
🎯 TP1: {round(s['tp1'],4)} | TP2: {round(s['tp2'],4)} | TP3: {round(s['tp3'],4)}

📊 4H: {s['trend_4h']} | 1H: {s['trend_1h']} | RSI: {s['rsi']}
🕯️ {', '.join(s['patterns_bull'] + s['patterns_bear']) or 'No pattern'}
"""

def fmt_multi_signals(signals, label, scanned, duration):
    if not signals:
        return (
            f"🎯 <b>{label} Complete</b>\n\n"
            f"Scanned: {scanned} coins\n"
            f"Duration: {duration}s\n\n"
            f"❌ No setups above min score {MIN_SCORE}"
        )

    header = (
        f"🎯 <b>{label} — Top {len(signals)} Signals</b>\n"
        f"Scanned: {scanned} coins | {duration}s\n"
        f"{'━' * 20}"
    )
    body = "".join(fmt_signal_block(s, i+1) for i, s in enumerate(signals))
    return header + body

# =========================================================
# TELEGRAM MENU
# =========================================================
def main_menu():
    kb = [
        [InlineKeyboardButton("⚡ Quick (5)", callback_data="quick"),
         InlineKeyboardButton("🎯 Deep (50)", callback_data="deep")],
        [InlineKeyboardButton("📊 Top 10", callback_data="top10"),
         InlineKeyboardButton("📈 Gainers", callback_data="gainers")],
        [InlineKeyboardButton("📉 Losers", callback_data="losers"),
         InlineKeyboardButton("🔍 Single Coin", callback_data="single")],
    ]
    return InlineKeyboardMarkup(kb)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Futures Signal Bot</b>\n\nChoose an option:\n\n"
        f"<i>Auto-scan runs every {AUTO_SCAN_MINUTES} minutes.</i>",
        reply_markup=main_menu(), parse_mode="HTML"
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    msg = q.message

    try:
        if data == "quick":
            await q.edit_message_text("⚡ Quick scan starting...")
            syms = (await get_top10())[:5]
            signals, dur = await run_scan(syms, msg, "Quick Scan", top_n=5)
            await q.edit_message_text(
                fmt_multi_signals(signals, "Quick Scan", len(syms), dur),
                parse_mode="HTML", reply_markup=main_menu())

        elif data == "deep":
            await q.edit_message_text("🎯 Deep analysis starting...")
            syms = await get_all_coins()
            signals, dur = await run_scan(syms, msg, "Deep Analysis", top_n=5)
            await q.edit_message_text(
                fmt_multi_signals(signals, "Deep Analysis", len(syms), dur),
                parse_mode="HTML", reply_markup=main_menu())

        elif data == "top10":
            await q.edit_message_text("📊 Scanning Top 10...")
            syms = await get_top10()
            signals, dur = await run_scan(syms, msg, "Top 10", top_n=5)
            await q.edit_message_text(
                fmt_multi_signals(signals, "Top 10", len(syms), dur),
                parse_mode="HTML", reply_markup=main_menu())

        elif data == "gainers":
            await q.edit_message_text("📈 Scanning Gainers...")
            syms = await get_gainers(20)
            signals, dur = await run_scan(syms, msg, "Gainers", top_n=5)
            await q.edit_message_text(
                fmt_multi_signals(signals, "Gainers", len(syms), dur),
                parse_mode="HTML", reply_markup=main_menu())

        elif data == "losers":
            await q.edit_message_text("📉 Scanning Losers...")
            syms = await get_losers(20)
            signals, dur = await run_scan(syms, msg, "Losers", top_n=5)
            await q.edit_message_text(
                fmt_multi_signals(signals, "Losers", len(syms), dur),
                parse_mode="HTML", reply_markup=main_menu())

        elif data == "single":
            await q.edit_message_text(
                "🔍 Send me a symbol:\n\nExample: <code>BTCUSDT</code>",
                parse_mode="HTML")
    except Exception as e:
        print(f"Button error: {e}")
        try:
            await q.edit_message_text(f"⚠️ Error: {str(e)[:100]}", reply_markup=main_menu())
        except Exception:
            pass

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sym = update.message.text.strip().upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    msg = await update.message.reply_text(f"🔍 Scanning {sym}...")
    try:
        result = await analyze_symbol(sym)
        if result and result["score"] >= MIN_SCORE:
            block = fmt_signal_block(result, 1)
            header = f"🎯 <b>{sym} — Single Coin</b>\n{'━' * 20}"
            await msg.edit_text(header + block, parse_mode="HTML", reply_markup=main_menu())
        else:
            score = result["score"] if result else 0
            await msg.edit_text(
                f"🔍 <b>{sym}</b>\n\nScore: {score}/30\n❌ Below min score {MIN_SCORE}",
                parse_mode="HTML", reply_markup=main_menu())
    except Exception as e:
        await msg.edit_text(f"⚠️ Error: {str(e)[:100]}", reply_markup=main_menu())

# =========================================================
# AUTO SCAN LOOP (every 30 min)
# =========================================================
async def auto_scan_loop(bot):
    await asyncio.sleep(60)  # startup delay
    while True:
        try:
            syms = await get_all_coins()
            print(f"[AUTO] Scanning {len(syms)} coins...")
            signals, dur = await run_scan(syms, None, "Auto Scan", top_n=5)
            if signals:
                text = fmt_multi_signals(signals, "Auto Scan", len(syms), dur)
                try:
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML")
                    print(f"[AUTO] Sent {len(signals)} signals")
                except Exception as e:
                    print(f"[AUTO] Send error: {e}")
            else:
                print("[AUTO] No signals this cycle")
        except Exception as e:
            print(f"[AUTO] Error: {e}")
        await asyncio.sleep(AUTO_SCAN_MINUTES * 60)

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
    print(f"✅ Health server on port {PORT}")

# =========================================================
# MAIN
# =========================================================
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    await start_web_server()
    print(f"🤖 Bot started (Bybit) — auto scan every {AUTO_SCAN_MINUTES} min")

    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🤖 <b>Bot is LIVE (Bybit)</b>\n\n"
                 f"Auto-scan every {AUTO_SCAN_MINUTES} min\n"
                 f"Send /start for menu",
            parse_mode="HTML")
    except Exception as e:
        print(f"Startup msg error: {e}")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    asyncio.create_task(auto_scan_loop(app.bot))

    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
