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

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df = df.set_index("timestamp")
    df = df.astype(float)

    return df


# ==========================================================
# CONFIG ETH, TRX, VET, UNI, DOGE, ETC, AAVE, BCH, OP, TIA, XLM, SUI,
# BTC, ZEN, AVAX, RUNE, ORDI, LDO, FIL, LINK, PENDLE
# ==========================================================

SYMBOL = "BTCUSDT"

LLTF_INTERVAL = "5m"
LTF_INTERVAL = "1h"
HTF_INTERVAL = "4h"

LEVERAGE = 1

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
# lltf_df = fetch_binance(SYMBOL, LLTF_INTERVAL, LLTF_LIMIT)

print("Downloading LTF data (1h)...")
ltf_df = fetch_binance(SYMBOL, LTF_INTERVAL, LTF_LIMIT)

print("Downloading HTF data (4h)...")
htf_df = fetch_binance(SYMBOL, HTF_INTERVAL, HTF_LIMIT)

# Drop the current incomplete 4H bar — live does this too
now_utc = pd.Timestamp.now(tz="UTC")

# Drop current incomplete 1H bar
current_1h_boundary = now_utc.floor("h")
ltf_df = ltf_df[ltf_df.index < current_1h_boundary].copy()

current_4h_boundary = now_utc.floor("4h")
htf_df = htf_df[htf_df.index < current_4h_boundary].copy()
print(f"[DEBUG] htf_df after incomplete bar drop: last={htf_df.index[-1]} len={len(htf_df)}")

# lltf_df.index = pd.to_datetime(lltf_df.index, utc=True)
ltf_df.index = pd.to_datetime(ltf_df.index, utc=True)
htf_df.index = pd.to_datetime(htf_df.index, utc=True)

# ltf_df.index = pd.to_datetime(ltf_df.index, utc=True)
# target = pd.Timestamp("2026-05-06 06:00:00", tz="UTC")
# print(f"Target in index: {target in ltf_df.index}")
# print(ltf_df.loc["2026-05-06 04:00":"2026-05-06 09:00"])

# ==========================================================
# SIGNAL GENERATION
# ==========================================================

ltf_df = generate_signal(ltf_df, htf_df)

# ==========================================================
# BACKTEST
# ==========================================================

backtester = SignalBacktester(ltf_df, htf_df=htf_df, leverage=LEVERAGE)

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

# ==========================================================
# HTF QUALITY DIAGNOSTIC — last 30 hours
# ==========================================================
print("\n=== HTF QUALITY (last 30 bars) ===")
print(f"{'timestamp':>25} {'HTF_DIR':>8} {'HTF_QUAL':>10} {'signal':>8} {'final_sig':>10}")
print("-" * 65)

diag_cols = ["HTF_DIRECTION", "HTF_QUALITY", "signal", "final_signal"]
available = [c for c in diag_cols if c in ltf_df.columns]

diag = ltf_df[available].tail(30)

for ts, row in diag.iterrows():
    htf_dir  = int(row["HTF_DIRECTION"])  if "HTF_DIRECTION"  in row.index else "N/A"
    htf_qual = f"{row['HTF_QUALITY']:.4f}" if "HTF_QUALITY"    in row.index else "N/A"
    sig      = int(row["signal"])          if "signal"          in row.index else "N/A"
    fsig     = int(row["final_signal"])    if "final_signal"    in row.index else "N/A"

    import pytz
    WAT = pytz.timezone("Africa/Lagos")
    ts_wat = ts.tz_convert(WAT).strftime("%Y-%m-%d %H:%M WAT")

    blocked = " ← BLOCKED" if (htf_qual != "N/A" and float(htf_qual) <= 0.45) else ""
    print(f"{ts_wat:>25} {str(htf_dir):>8} {htf_qual:>10} {str(sig):>8} {str(fsig):>10}{blocked}")

print(f"\nHTF threshold: 0.45")
print(f"Last HTF_DIRECTION : {int(ltf_df['HTF_DIRECTION'].iloc[-1])}")
print(f"Last HTF_QUALITY   : {ltf_df['HTF_QUALITY'].iloc[-1]:.4f}")
print(f"Last final_signal  : {int(ltf_df['final_signal'].iloc[-1])}")

# # ── 5m candle dump per trade ────────────────────────────────
# print("\n=== 5M CANDLES PER TRADE ===")
# for _, t in trade_log.iterrows():
#     entry_time = backtester.lltf_df.index[int(t["entry_idx"])]
#     exit_time  = backtester.lltf_df.index[int(t["exit_idx"])]
#     window     = backtester.lltf_df.loc[entry_time:exit_time].copy()
#     R          = abs(t["entry_price"] - t["stop_loss"])
#     print(f"\n{'='*60}")
#     print(f"{t['direction']} | entry={t['entry_price']:.4f} stop={t['stop_loss']:.4f} R={R:.4f}")
#     print(f"{'time':>8} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'vol':>12} {'body/atr':>9} {'stop_r':>7} {'pnl_r':>7}")
#     for ts, row in window.iterrows():
#         body    = abs(row["close"] - row["open"])
#         atr_5m  = row.get("ATR_5M", float("nan"))
#         if pd.isna(atr_5m) or atr_5m <= 0:
#             atr_5m = row.get("ATR", float("nan")) * 0.20
#         body_atr = body / atr_5m if atr_5m and not pd.isna(atr_5m) and atr_5m > 0 else float("nan")
#         if t["side"] == 1:
#             stop_r = (row["close"] - t["stop_loss"]) / R if R > 0 else float("nan")
#             pnl_r  = (row["close"] - t["entry_price"]) / R if R > 0 else float("nan")
#         else:
#             stop_r = (t["stop_loss"] - row["close"]) / R if R > 0 else float("nan")
#             pnl_r  = (t["entry_price"] - row["close"]) / R if R > 0 else float("nan")
#         wat_ts = (ts + pd.Timedelta(hours=1)).strftime("%H:%M")
#         print(f"{wat_ts:>8} {row['open']:>8.4f} {row['high']:>8.4f} {row['low']:>8.4f} {row['close']:>8.4f} {row['volume']:>12.2f} {body_atr:>9.2f} {stop_r:>7.2f} {pnl_r:>7.2f}")