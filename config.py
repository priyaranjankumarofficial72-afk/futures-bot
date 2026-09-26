import os

# =========================================================
# FUTURES SIGNAL BOT — CONFIG (70 Factors)
# =========================================================

# ---------- TELEGRAM ----------
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "PASTE_YOUR_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PASTE_YOUR_CHAT_ID")

# ---------- BINANCE ----------
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "PASTE_YOUR_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "PASTE_YOUR_SECRET_KEY")

# ---------- NEWS (optional) ----------
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")

# ---------- SCANNING ----------
SCAN_TOP10_BY_VOLUME = True
SCAN_TOP_GAINERS = 20
SCAN_TOP_LOSERS = 20
MIN_24H_VOLUME_USD = 10_000_000
SCAN_INTERVAL_SECONDS = 3600   # 1 hour

# ---------- SIGNAL OUTPUT ----------
MAX_SIGNALS_PER_CYCLE = 1
SEND_NO_TRADE_REPORT = False

# ---------- RISK ----------
ACCOUNT_BALANCE = 100.0
RISK_PER_TRADE = 0.01
MIN_RR = 2.0
LEVERAGE = 5
MAX_DAILY_LOSS = 0.03

# ---------- FACTOR MODE ----------
FACTOR_MODE = 70   # options: 30 (fast) | 70 (balanced) | 150 (extreme)

# ---------- WEIGHTED SCORING ----------
MIN_SCORE = 30   # out of ~70 max — sends signal only if score >= this

# ---------- RATE LIMIT SAFETY ----------
FETCH_BATCH_SIZE = 10
BATCH_DELAY_SECONDS = 3
