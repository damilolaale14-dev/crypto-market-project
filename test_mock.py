"""
test_oie.py
-----------
Reconstructs what lifecycle.py _opposite_impulse_exit would have seen
bar by bar for the BANDUSDT SHORT trade on 2026-05-20.

Run from your project root:
    python test_oie.py

Requires: data/backtest_cache/BANDUSDT_5m.parquet
"""

import pandas as pd
import numpy as np

# ── TRADE CONSTANTS ──────────────────────────────────────────────────────────
SYMBOL        = "BANDUSDT"
SIDE          = -1           # SHORT
ENTRY_PRICE   = 0.205000
INITIAL_STOP  = 0.207661
ENTRY_TIME    = pd.Timestamp("2026-05-20 16:00:00", tz="UTC")
R             = abs(ENTRY_PRICE - INITIAL_STOP)

# ── LOAD 5M DATA ─────────────────────────────────────────────────────────────
path = "data/backtest_cache/BANDUSDT_5m.parquet"
df = pd.read_parquet(path)
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()

# Compute ATR_5M the same way hourly_runner does
def atr_ema(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

df['ATR_5M'] = atr_ema(df, period=14)

# Also forward-fill 1H ATR onto 5m bars the same way hourly_runner does
# We approximate by computing ATR on the 5m data at 1H scale
# (in reality this comes from the 1H df reindexed — but for the body/atr
#  comparison what matters is the magnitude, and ATR_5M is what the exit uses)

# ── SLICE THE TRADE WINDOW ───────────────────────────────────────────────────
# Start 10 bars before entry so volume baseline has history
pre_entry_bars = 10
entry_idx = df.index.searchsorted(ENTRY_TIME)
start_idx = max(0, entry_idx - pre_entry_bars)

window_full = df.iloc[start_idx:entry_idx + 40].copy()  # 40 bars after entry

print(f"Entry time : {ENTRY_TIME}")
print(f"Entry price: {ENTRY_PRICE}")
print(f"Stop       : {INITIAL_STOP}")
print(f"R          : {R:.6f}")
print(f"Bars loaded: {len(window_full)}")
print(f"Window     : {window_full.index[0]} → {window_full.index[-1]}")
print()

# ── SIMULATE _bar_history GROWTH ─────────────────────────────────────────────
# lifecycle.py resets _bar_history to just the entry bar at open,
# then appends one bar at a time. We simulate that exactly.

def opposite_impulse_exit(window_df, side, position, bar_num):
    """
    Exact copy of lifecycle.py _opposite_impulse_exit with print statements.
    """
    if len(window_df) < 3:
        print(f"  [OIE] bar={bar_num:>3} | SKIP — window too short ({len(window_df)})")
        return False

    last = window_df.iloc[-1]

    # 1. ATR
    if "ATR_5M" in window_df.columns:
        atr = window_df["ATR_5M"].iloc[-3:].mean()
        if pd.isna(atr) or atr <= 0:
            atr = window_df["ATR_5M"].iloc[0]
    else:
        atr = None

    if atr is None or pd.isna(atr) or atr <= 0:
        atr_1h = window_df["ATR"].iloc[-3:].mean() if "ATR" in window_df.columns else np.nan
        if pd.isna(atr_1h) or atr_1h <= 0:
            print(f"  [OIE] bar={bar_num:>3} | SKIP — no valid ATR")
            return False
        atr = atr_1h * 0.20

    # 2. Body size
    body = abs(last["close"] - last["open"])
    big_candle = body > atr * 1.2

    # 3. Direction
    if side == 1:
        wrong_direction = last["close"] < last["open"]
    else:
        wrong_direction = last["close"] > last["open"]

    # 4. Close location
    entry  = position["entry_price"]
    stop   = position["stop_loss"]
    R_val  = abs(entry - position["initial_stop"])
    if R_val == 0:
        close_to_stop_r = 0.0
    else:
        if side == 1:
            close_to_stop_r = (last["close"] - stop) / R_val
        else:
            close_to_stop_r = (stop - last["close"]) / R_val

    mfe_r = position.get("mfe_r", 0.0)
    location_blocked = close_to_stop_r > 1.5 and mfe_r < 1.0

    # 5. Volume
    vol_blocked = False
    avg_vol = np.nan
    last_vol = last.get("volume", np.nan)
    if "volume" in window_df.columns and len(window_df) >= 10:
        avg_vol = window_df["volume"].iloc[-10:].mean()
        if not pd.isna(avg_vol) and avg_vol > 0:
            if last_vol < avg_vol * 0.8:
                vol_blocked = True

    # ── RESULT ────────────────────────────────────────────────────────────────
    fired = big_candle and wrong_direction and not location_blocked and not vol_blocked

    status = "🔥 FIRED" if fired else "  miss "
    print(
        f"  [OIE] bar={bar_num:>3} | {status} | "
        f"ts={last.name} | "
        f"o={last['open']:.5f} c={last['close']:.5f} | "
        f"body={body:.5f} atr={atr:.5f} thr={atr*1.2:.5f} big={big_candle} | "
        f"wrong_dir={wrong_direction} | "
        f"csr={close_to_stop_r:.3f} loc_block={location_blocked} | "
        f"vol_block={vol_blocked} (last={last_vol:.0f} avg={avg_vol:.0f} wlen={len(window_df)})"
    )

    return fired


# ── RUN SIMULATION ────────────────────────────────────────────────────────────
# Find entry bar index in the sliced window
entry_bar_idx = window_full.index.searchsorted(ENTRY_TIME)
pre_entry     = window_full.iloc[:entry_bar_idx]   # bars before entry
post_entry    = window_full.iloc[entry_bar_idx:]    # entry bar onwards

print("=" * 100)
print("SIMULATING _bar_history + _opposite_impulse_exit BAR BY BAR")
print("=" * 100)

# Initialise _bar_history with just the entry bar (as lifecycle.py does)
# NOTE: no ATR_5M in the initial entry bar — this matches the bug we suspect
entry_bar = post_entry.iloc[0]
bar_history = [{
    "open":   float(entry_bar["open"]),
    "high":   float(entry_bar["high"]),
    "low":    float(entry_bar["low"]),
    "close":  float(entry_bar["close"]),
    "ATR":    float(entry_bar.get("ATR_5M", 0.0)),   # lifecycle uses 1H ATR here, no ATR_5M
    "ATR_5M": float(entry_bar["ATR_5M"]) if not pd.isna(entry_bar["ATR_5M"]) else None,
    "volume": float(entry_bar["volume"]),
    "ts":     entry_bar.name,
}]

# Track MFE
mfe = 0.0
position = {
    "entry_price":  ENTRY_PRICE,
    "stop_loss":    INITIAL_STOP,
    "initial_stop": INITIAL_STOP,
    "direction":    SIDE,
    "mfe_r":        0.0,
    "MFE":          0.0,
}

print(f"\nEntry bar: {entry_bar.name} | o={entry_bar['open']:.5f} h={entry_bar['high']:.5f} "
      f"l={entry_bar['low']:.5f} c={entry_bar['close']:.5f}")
print()

fired_at = None
for bar_num, (ts, row) in enumerate(post_entry.iloc[1:].iterrows(), start=1):

    # Append bar to history (same as lifecycle.py)
    atr_5m_val = float(row["ATR_5M"]) if "ATR_5M" in row.index and not pd.isna(row["ATR_5M"]) else None
    bar_history.append({
        "open":   float(row["open"]),
        "high":   float(row["high"]),
        "low":    float(row["low"]),
        "close":  float(row["close"]),
        "ATR":    float(row.get("ATR_5M", 0.0)),
        "ATR_5M": atr_5m_val,
        "volume": float(row["volume"]),
        "ts":     ts,
    })

    window_df = pd.DataFrame(bar_history)

    # Update MFE
    if SIDE == -1:
        move = ENTRY_PRICE - float(row["low"])
    else:
        move = float(row["high"]) - ENTRY_PRICE
    mfe = max(mfe, move)
    position["MFE"]   = mfe
    position["mfe_r"] = mfe / R

    # Update stop for reporting
    pnl_r = (ENTRY_PRICE - float(row["close"])) / R if SIDE == -1 else (float(row["close"]) - ENTRY_PRICE) / R

    fired = opposite_impulse_exit(window_df, SIDE, position, bar_num)
    if fired and fired_at is None:
        fired_at = bar_num
        print(f"\n  ^^^ FIRST FIRE at bar {bar_num} ({ts}) — would have exited here\n")

    # Hard stop check
    if SIDE == -1 and float(row["high"]) >= INITIAL_STOP:
        print(f"\n  ⛔ HARD STOP HIT at bar {bar_num} ({ts}) | high={row['high']:.5f} >= stop={INITIAL_STOP:.5f}\n")
        break

print()
print("=" * 100)
if fired_at:
    print(f"RESULT: Opposite impulse would have fired at bar {fired_at}")
else:
    print("RESULT: Opposite impulse NEVER fired in this window — exit logic broken or market too flat")
print("=" * 100)

# ── BONUS: print the raw 5m bars for the trade window ────────────────────────
print("\nRAW 5M BARS (entry onwards):")
print(f"{'ts':>32} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'volume':>12} {'ATR_5M':>10} {'pnl_r':>8}")
for ts, row in post_entry.iterrows():
    pnl_r = (ENTRY_PRICE - float(row["close"])) / R if SIDE == -1 else (float(row["close"]) - ENTRY_PRICE) / R
    print(
        f"  {str(ts):>30} "
        f"{row['open']:>8.5f} {row['high']:>8.5f} {row['low']:>8.5f} {row['close']:>8.5f} "
        f"{row['volume']:>12.0f} {row['ATR_5M']:>10.6f} {pnl_r:>8.3f}R"
    )