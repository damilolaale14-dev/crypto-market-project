import pandas as pd
import requests
import matplotlib.pyplot as plt
import time

from indicators.indicators import generate_signal
from backtest import SignalBacktester
from trade_diagnostics import diagnose_trades
from diagnostics import plot_asymmetry


# ==========================================================
# BINANCE DATA FETCHER
# ==========================================================

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_binance(symbol, interval, limit):

    all_data = []
    end_time = None

    while len(all_data) < limit:

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(1000, limit - len(all_data))
        }

        if end_time:
            params["endTime"] = end_time

        response = requests.get(BINANCE_URL, params=params)
        data = response.json()

        if not data:
            break

        all_data = data + all_data

        first_open_time = data[0][0]
        end_time = first_open_time - 1
        time.sleep(0.25)

        if len(data) < 1000:
            break

    df = pd.DataFrame(all_data, columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ])

    df = df[["open_time", "open", "high", "low", "close", "volume", "taker_buy_base"]]

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.drop(columns=["open_time"])
    df = df.set_index("timestamp")
    df = df.astype(float)

    return df


# ==========================================================
# CONFIG ETH, FIL, TRX, VET, UNI, DOGE, ETC, AAVE, BCH, BAND, TIA, XLM, SUI, 
# BTC, ZEN, AVAX, AXS, ORDI, LDO, LINK
# ==========================================================

SYMBOL = "BANDUSDT"

LLTF_INTERVAL = "5m"
LTF_INTERVAL = "1h"
HTF_INTERVAL = "4h"

# LLTF_LIMIT = 630720
# LTF_LIMIT = 52560   # ~30 days of 1h candles
# HTF_LIMIT = 13140   # ~120 days of 4h candles

# LTF_LIMIT = 43800   # ~30 days of 1h candles
# HTF_LIMIT = 10950   # ~120 days of 4h candles

# LTF_LIMIT = 35040   # ~30 days of 1h candles
# HTF_LIMIT = 8760   # ~120 days of 4h candles

# LTF_LIMIT = 26280   # ~30 days of 1h candles
# HTF_LIMIT = 6570   # ~120 days of 4h candles

# LLTF_LIMIT = 210240
# LTF_LIMIT = 17520   # ~30 days of 1h candles
# HTF_LIMIT = 4380   # ~120 days of 4h candles

# LTF_LIMIT = 8760   # ~30 days of 1h candles
# HTF_LIMIT = 2190   # ~120 days of 4h candles

# LLTF_LIMIT = 52560
# LTF_LIMIT = 4380   # ~30 days of 1h candles
# HTF_LIMIT = 1095   # ~120 days of 4h candles

# LLTF_LIMIT = 24000
# LTF_LIMIT = 2000   # ~30 days of 1h candles
# HTF_LIMIT = 500   # ~120 days of 4h candles

LLTF_LIMIT = 12000
LTF_LIMIT = 1000   # ~30 days of 1h candles
HTF_LIMIT = 250   # ~120 days of 4h candles

# LLTF_LIMIT = 6000
# LTF_LIMIT = 500   # ~30 days of 1h candles
# HTF_LIMIT = 125   # ~120 days of 4h candles

# ==========================================================
# FETCH DATA
# ==========================================================

print("Downloading LLTF data (5m)...")
lltf_df = fetch_binance(SYMBOL, LLTF_INTERVAL, LLTF_LIMIT)

print("Downloading LTF data (1h)...")
ltf_df = fetch_binance(SYMBOL, LTF_INTERVAL, LTF_LIMIT)

print("Downloading HTF data (4h)...")
htf_df = fetch_binance(SYMBOL, HTF_INTERVAL, HTF_LIMIT)

# ==========================================================
# SIGNAL GENERATION
# ==========================================================

ltf_df = generate_signal(ltf_df, htf_df)

# ==========================================================
# BACKTEST
# ==========================================================

backtester = SignalBacktester(ltf_df, htf_df=htf_df, lltf_df=lltf_df)

backtest_output = backtester.run()

trade_log = backtest_output["trades"]
equity_curve = backtest_output["equity_curve"]
results = backtest_output["summary"]

print(results)

print("=== TRADE LOG ===")
print(trade_log.head(10))

print("\nColumns:", trade_log.columns)
print("\nNumber of trades:", len(trade_log))
print("LTF candles (1h):", len(ltf_df))
print("HTF candles (4h):", len(htf_df))

# ==========================================================
# DIAGNOSTICS
# ==========================================================

diagnostics_df = diagnose_trades(trade_log)