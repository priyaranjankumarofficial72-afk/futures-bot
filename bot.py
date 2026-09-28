# =========================================================
# FUTURES SIGNAL BOT — Simplified (15 Factors + 15 Patterns)
# Public Binance Data Endpoint (no IP bans)
# =========================================================

import asyncio
import csv
import os
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import aiohttp
from aiohttp import web
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# =========================================================
# CONFIG
# =========================================================
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MIN_VOLUME_USD = 10_000_000
MIN_SCORE = 12
PORT = int(os.environ.get("PORT", 10000))

API_BASE = "https://fapi.binance.com"

# =========================================================
# HTTP HELPERS
# =========================================================
async def http_get(url, timeout=10):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
    except Exception as e:
        print(f"HTTP error {url}: {e}")
        return None

async def fetch_klines(symbol, interval, limit=250):
    url = f"{API_BASE}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = await http_get(url)
    if not data:
        return None
    try:
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        print(f"Klines parse error {symbol}: {e}")
        return None

async def fetch_ticker():
    return await http_get(f"{API_BASE}/fapi/v1/ticker/24hr")

def body(c): return abs(c["close"] - c["open"])
def rng(c):  return c["high"] - c["low"]
def upper_wick(c): return c["high"] - max(c["open"], c["close"])
def lower_wick(c): return min(c["open"], c["close"]) - c["low"]
def is_bull(c): return c["close"] > c["open"]
def is_bear(c): return c["close"] < c["open"]
def mid(c): return (c["open"] + c["close"]) / 2

# =========================================================
# INDICATORS
# =========================================================
def add_indicators(df):
    try:
        df["ema50"]  = EMAIndicator(df["close"], 50).ema_indicator()
        df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()
        df["rsi"]    = RSIIndicator(df["close"], 14).rsi()
        macd = MACD(df["close"])
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df["vol_avg"] = df["volume"].rolling(20).mean()
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
        hh = sh[-1] > sh[-2]; hl = sl[-1] > sl[-2]
        lh = sh[-1] < sh[-2]; ll = sl[-1] < sl[-2]
        if hh and hl: return "HH_HL"
        if lh and ll: return "LH_LL"
    return "UNCLEAR"

def get_key_levels(df_daily):
    if df_daily is None or len(df_daily) < 2:
        return {"pdh": None, "pdl": None}
    prev = df_daily.iloc[-2]
    return {"pdh": prev["high"], "pdl": prev["low"]}

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
# CANDLESTICK PATTERNS (15 essential)
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
def is_three_inside_up(c1, c2, c3):
    return (is_bear(c1) and is_bull(c2) and c2["open"] > c1["close"] and
            c2["close"] < c1["open"] and is_bull(c3) and c3["close"] > c1["open"])
def is_three_inside_down(c1, c2, c3):
    return (is_bull(c1) and is_bear(c2) and c2["open"] < c1["close"] and
            c2["close"] > c1["open"] and is_bear(c3) and c3["close"] < c1["open"])

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
    if is_three_inside_up(c3, c4, c5): p.append("THREE_INSIDE_UP")
    if is_three_inside_down(c3, c4, c5): p.append("THREE_INSIDE_DOWN")
    return p

BULLISH_P = {"HAMMER","INVERTED_HAMMER","BULLISH_ENGULFING","PIERCING_LINE",
             "TWEEZER_BOTTOM","MORNING_STAR","THREE_WHITE_SOLDIERS",
             "THREE_INSIDE_UP","BULLISH_MARUBOZU"}
BEARISH_P = {"SHOOTING_STAR","BEARISH_ENGULFING","DARK_CLOUD_COVER",
             "TWEEZER_TOP","EVENING_STAR","THREE_BLACK_CROWS",
             "THREE_INSIDE_DOWN","BEARISH_MARUBOZU"}

# =========================================================
# SCORING
# =========================================================
def compute_score(s):
    score = 0
    if s["trend_4h"] in ("BULLISH","BEARISH"): score += 2
    if s["trend_1h"] in ("BULLISH","BEARISH"): score += 2
    if s["structure"] in ("HH_HL","LH_LL"): score += 2
    if s["key_level"]: score += 2
    if s["sweep"]: score += 3
    if s["volume"]: score += 3
    if s["vwap_ok"]: score += 1
    if s["rsi_ok"]: score += 1
    if s["macd_ok"]: score += 1
    if s["patterns_bull"]: score += 3
    if s["patterns_bear"]: score += 3
    return score

# =========================================================
# MASTER ANALYZER
# =========================================================
async def analyze_symbol(symbol):
    try:
        df_4h  = add_indicators(await fetch_klines(symbol, "4h"))
        df_1h  = add_indicators(await fetch_klines(symbol, "1h"))
        df_15m = add_indicators(await fetch_klines(symbol, "15m"))
        df_1d  = await fetch_klines(symbol, "1d", 5)
        if any(x is None for x in [df_4h, df_1h, df_15m, df_1d]):
            return None
    except Exception as e:
        print(f"analyze_symbol error {symbol}: {e}")
        return None

    trend_4h = get_trend(df_4h)
    trend_1h = get_trend(df_1h)
    structure = get_structure(df_1h)

    levels = get_key_levels(df_1d)
    price = df_15m.iloc[-1]["close"]
    key_level = near_key_level_local(price, levels)
    sweep = check_liquidity_sweep(df_15m, levels)

    vol_ok = df_15m.iloc[-1]["volume"] > df_15m.iloc[-1]["vol_avg"] * 1.5
    rsi = df_15m.iloc[-1]["rsi"]
    macd_now = df_15m.iloc[-1]["macd"]
    macd_sig = df_15m.iloc[-1]["macd_signal"]
    atr = df_15m.iloc[-1]["atr"]
    vwap = df_15m.iloc[-1]["vwap"]

    patterns_15m = detect_patterns(df_15m)
    bull_p = [p for p in patterns_15m if p in BULLISH_P]
    bear_p = [p for p in patterns_15m if p in BEARISH_P]

    base = {
        "symbol": symbol, "price": price,
        "trend_4h": trend_4h, "trend_1h": trend_1h,
        "structure": structure, "key_level": key_level,
        "sweep": sweep, "volume": vol_ok,
        "rsi": round(rsi, 2), "atr": round(atr, 4), "vwap": round(vwap, 4),
        "patterns_bull": bull_p, "patterns_bear": bear_p,
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
            "rsi_ok": rsi < 65, "macd_ok": True, "vwap_ok": True,
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
            "rsi_ok": rsi > 35, "macd_ok": True, "vwap_ok": True,
        })
        base["score"] = compute_score(base)
        return base

    return None

def near_key_level_local(price, levels, tol=0.003):
    for name, lvl in levels.items():
        if lvl is None:
            continue
        if abs(price - lvl) / lvl < tol:
            return name
    return None
    # =========================================================
# SCANNER
# =========================================================
async def get_top10():
    tickers = await fetch_ticker()
    if not tickers:
        return []
    perps = [t for t in tickers if t["symbol"].endswith("USDT")
             and not any(x in t["symbol"] for x in ["_","UP","DOWN","BULL","BEAR"])]
    perps.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in perps[:10]]

async def get_gainers(n=20):
    tickers = await fetch_ticker()
    if not tickers:
        return []
    clean = []
    for t in tickers:
        s = t["symbol"]
        if not s.endswith("USDT"): continue
        if any(x in s for x in ["_","UP","DOWN","BULL","BEAR"]): continue
        try:
            if float(t["quoteVolume"]) < MIN_VOLUME_USD: continue
            clean.append({"symbol": s, "pct": float(t["priceChangePercent"])})
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
        if any(x in s for x in ["_","UP","DOWN","BULL","BEAR"]): continue
        try:
            if float(t["quoteVolume"]) < MIN_VOLUME_USD: continue
            clean.append({"symbol": s, "pct": float(t["priceChangePercent"])})
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
async def run_scan(symbols, update_msg, label):
    start = time.time()
    best = None
    total = len(symbols)

    for i, sym in enumerate(symbols):
        elapsed = int(time.time() - start)
        remaining = int((elapsed / i) * (total - i)) if i > 0 else 0

        if i % 2 == 0:
            pct = i * 10 // max(total, 1)
            bar = "█" * pct + "░" * (10 - pct)
            best_txt = f"Best: {best['score']} ({best['symbol']})" if best else "Best: searching..."
            txt = (
                f"🎯 <b>{label}</b>\n\n"
                f"Progress: [{bar}] {i}/{total}\n"
                f"Current: {sym}\n"
                f"Elapsed: {elapsed}s | Left: ~{remaining}s\n"
                f"{best_txt}"
            )
            try:
                await update_msg.edit_text(txt, parse_mode="HTML")
            except Exception:
                pass

        result = await analyze_symbol(sym)
        if result and result.get("score", 0) > (best["score"] if best else 0):
            best = result
        await asyncio.sleep(0.3)

    return best, int(time.time() - start)

# =========================================================
# MESSAGE FORMAT
# =========================================================
def fmt_signal(s):
    emoji = "🟢" if s["side"] == "LONG" else "🔴"
    return f"""
{emoji} <b>{s['symbol']} — {s['side']}</b>
🎯 Score: <b>{s['score']}</b>

📊 <b>CONTEXT</b>
• 4H Trend: {s['trend_4h']}
• 1H Trend: {s['trend_1h']}
• Structure: {s['structure']}
• Sweep: {s['sweep'] or '-'}
• Key Level: {s['key_level'] or '-'}

🕯️ <b>PATTERNS</b>
• Bull: {', '.join(s['patterns_bull']) or '-'}
• Bear: {', '.join(s['patterns_bear']) or '-'}

📈 <b>INDICATORS</b>
• RSI: {s['rsi']}
• ATR: {s['atr']}
• VWAP: {s['vwap']}
• Volume: {'✅' if s['volume'] else '❌'}

💰 <b>TRADE PLAN</b>
• Entry: {round(s['entry'],4)}
• Stop-Loss: {round(s['sl'],4)}
• TP1: {round(s['tp1'],4)}
• TP2: {round(s['tp2'],4)}
• TP3: {round(s['tp3'],4)}

⚖️ R:R = 1:{s['rr']}
⏰ {datetime.now(timezone.utc).strftime('%H:%M')} UTC
"""

def fmt_no_signal(label, scanned, duration, best):
    best_txt = f"Best: {best['score']} ({best['symbol']})" if best else "No score"
    return (
        f"🎯 <b>{label} Complete</b>\n\n"
        f"Scanned: {scanned} coins\n"
        f"Duration: {duration}s\n"
        f"{best_txt}\n\n"
        f"❌ No setup above min score {MIN_SCORE}"
    )

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
        "🤖 <b>Futures Signal Bot</b>\n\nChoose an option:",
        reply_markup=main_menu(),
        parse_mode="HTML"
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
            best, dur = await run_scan(syms, msg, "Quick Scan")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML", reply_markup=main_menu())
            else:
                await q.edit_message_text(fmt_no_signal("Quick Scan", len(syms), dur, best), parse_mode="HTML", reply_markup=main_menu())

        elif data == "deep":
            await q.edit_message_text("🎯 Deep analysis starting...")
            syms = await get_all_coins()
            best, dur = await run_scan(syms, msg, "Deep Analysis")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML", reply_markup=main_menu())
            else:
                await q.edit_message_text(fmt_no_signal("Deep Analysis", len(syms), dur, best), parse_mode="HTML", reply_markup=main_menu())

        elif data == "top10":
            await q.edit_message_text("📊 Scanning Top 10...")
            syms = await get_top10()
            best, dur = await run_scan(syms, msg, "Top 10 Scan")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML", reply_markup=main_menu())
            else:
                await q.edit_message_text(fmt_no_signal("Top 10", len(syms), dur, best), parse_mode="HTML", reply_markup=main_menu())

        elif data == "gainers":
            await q.edit_message_text("📈 Scanning Gainers...")
            syms = await get_gainers(20)
            best, dur = await run_scan(syms, msg, "Gainers Scan")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML", reply_markup=main_menu())
            else:
                await q.edit_message_text(fmt_no_signal("Gainers", len(syms), dur, best), parse_mode="HTML", reply_markup=main_menu())

        elif data == "losers":
            await q.edit_message_text("📉 Scanning Losers...")
            syms = await get_losers(20)
            best, dur = await run_scan(syms, msg, "Losers Scan")
            if best and best["score"] >= MIN_SCORE:
                await q.edit_message_text(fmt_signal(best), parse_mode="HTML", reply_markup=main_menu())
            else:
                await q.edit_message_text(fmt_no_signal("Losers", len(syms), dur, best), parse_mode="HTML", reply_markup=main_menu())

        elif data == "single":
            await q.edit_message_text(
                "🔍 Send me a symbol:\n\nExample: <code>BTCUSDT</code>",
                parse_mode="HTML"
            )
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
            await msg.edit_text(fmt_signal(result), parse_mode="HTML", reply_markup=main_menu())
        else:
            score = result["score"] if result else 0
            await msg.edit_text(
                f"🔍 <b>{sym}</b>\n\nScore: {score}\n❌ Below min score {MIN_SCORE}",
                parse_mode="HTML", reply_markup=main_menu())
    except Exception as e:
        await msg.edit_text(f"⚠️ Error: {str(e)[:100]}", reply_markup=main_menu())

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
    print("🤖 Bot started")

    try:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text="🤖 <b>Bot is LIVE</b>\n\nSend /start",
            parse_mode="HTML"
        )
    except Exception as e:
        print(f"Startup msg error: {e}")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
