"""
test_reentry_live.py
====================
Run from project root: python test_reentry_live.py

What this does:
  1. Downloads real Binance candles for BTCUSDT (1h, 4h, 5m)
  2. Patches TelegramNotifier so nothing actually sends
  3. Runs PositionManager directly, feeding bars one at a time
  4. Forces a trade OPEN on the first signal bar found
  5. Immediately forces it CLOSED via opposite_impulse on the next bar
  6. Then keeps feeding bars and watches whether the system re-enters
     the same direction — which it should NOT do within the same hour

All output goes to terminal. State is written to data/test_reentry/
so it never touches your real data/positions/ files.

At the end it prints a clear PASS or FAIL verdict with the full
sequence of events so you can see exactly where the bug is.
"""

import os
import sys
import json
import types
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────
# 0. REDIRECT STATE FILES TO TEST SANDBOX
# ─────────────────────────────────────────────
TEST_DIR = "data/test_reentry"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR, exist_ok=True)

# Patch the constants in lifecycle BEFORE importing it
import strategy.lifecycle as lifecycle_module
lifecycle_module.POSITIONS_DIR       = TEST_DIR
lifecycle_module.POSITIONS_FILE      = os.path.join(TEST_DIR, "open_positions.json")
lifecycle_module.BAR_HISTORY_FILE    = os.path.join(TEST_DIR, "bar_history.json")
lifecycle_module.ENTRY_TS_FILE       = os.path.join(TEST_DIR, "last_entry_ts.json")
lifecycle_module.EXECUTED_SIGNALS_FILE = os.path.join(TEST_DIR, "executed_signals.json")
lifecycle_module.REENTRY_LOCK_FILE   = os.path.join(TEST_DIR, "reentry_lock.json")

# ─────────────────────────────────────────────
# 1. MOCK TELEGRAM — print instead of send
# ─────────────────────────────────────────────
class MockNotifier:
    """Replaces TelegramNotifier. Prints everything to terminal."""

    @staticmethod
    def _fmt_ts(ts):
        try:
            return pd.Timestamp(ts).strftime("%H:%M WAT")
        except Exception:
            return str(ts)

    @staticmethod
    def make_trade_id(symbol):
        import uuid
        return f"{symbol}-{uuid.uuid4().hex[:8]}"

    def send_text(self, msg):
        print(f"[TG] {msg}")

    def debug(self, msg):
        print(f"[TG:debug] {msg}")

    def notify_open(self, **kwargs):
        print(
            f"\n  🟢 TRADE OPENED\n"
            f"     symbol={kwargs.get('symbol')} "
            f"dir={kwargs.get('direction')} "
            f"entry={kwargs.get('entry_price')} "
            f"stop={kwargs.get('stop_loss'):.6f} "
            f"risk=${kwargs.get('risk_usd')}"
        )

    def notify_close(self, **kwargs):
        print(
            f"\n  🔴 TRADE CLOSED\n"
            f"     symbol={kwargs.get('symbol')} "
            f"reason={kwargs.get('reason')} "
            f"pnl={kwargs.get('pnl_r'):+.3f}R"
        )


# Patch notifier everywhere it's used
lifecycle_module.TelegramNotifier = MockNotifier

# Also patch it in the execution.notifier module if already imported
try:
    import execution.notifier as notifier_mod
    notifier_mod.TelegramNotifier = MockNotifier
except Exception:
    pass

from strategy.lifecycle import PositionManager

# ─────────────────────────────────────────────
# 2. FETCH REAL CANDLES
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FETCHING REAL BINANCE CANDLES FOR BTCUSDT")
print("="*60)

from data_pipeline.updater import update_symbol

SYMBOL = "BTCUSDT"

print(f"  Fetching 1h / 4h / 5m candles for {SYMBOL}...")
try:
    df_1h, df_4h, df_5m = update_symbol(SYMBOL)
    print(f"  ✅ 1h candles : {len(df_1h)}  ({df_1h.index[0]} → {df_1h.index[-1]})")
    print(f"  ✅ 4h candles : {len(df_4h)}  ({df_4h.index[0]} → {df_4h.index[-1]})")
    print(f"  ✅ 5m candles : {len(df_5m)}  ({df_5m.index[0]} → {df_5m.index[-1]})")
except Exception as e:
    print(f"  ❌ fetch failed: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# 3. GENERATE SIGNALS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("GENERATING SIGNALS")
print("="*60)

from indicators.indicators import generate_signal, atr_ema
from execution.hourly_runner import map_ltf_to_htf

df_1h_sig = generate_signal(df_1h.copy(), df_4h.copy(), live=True)

# Map 5m bars to their parent 1h bar
df_5m_mapped = map_ltf_to_htf(df_5m.copy(), df_1h_sig)
df_5m_mapped["final_signal"] = df_1h_sig["final_signal"].reindex(
    df_5m_mapped.index, method="ffill"
)
df_5m_mapped["ATR"]    = df_1h_sig["ATR"].reindex(df_5m_mapped.index, method="ffill")
df_5m_mapped["ATR_5M"] = atr_ema(df_5m_mapped, period=14)

# Drop NaN ltf_index rows
df_5m_mapped = df_5m_mapped.dropna(subset=["ltf_index"])
df_5m_mapped["ltf_index"] = df_5m_mapped["ltf_index"].astype(int)

# Find bars with a signal
signal_bars = df_5m_mapped[df_5m_mapped["final_signal"] != 0]
print(f"  Total 5m bars with signal : {len(signal_bars)}")
if signal_bars.empty:
    print("  ❌ No signal bars found — cannot run test")
    sys.exit(1)

# Pick the most recent signal bar to test with
test_bar   = signal_bars.iloc[-1]
test_ts    = test_bar.name
test_sig   = int(test_bar["final_signal"])
test_1h_row = df_1h_sig.iloc[int(test_bar["ltf_index"])]

print(f"  Using signal bar : {test_ts}  signal={test_sig}")
print(f"  Parent 1h bar   : {test_1h_row.name}")

# ─────────────────────────────────────────────
# 4. SIMULATE BAR-BY-BAR
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING BAR-BY-BAR SIMULATION")
print("="*60)

pm = PositionManager(persist=True, notify=True)

# Grab bars starting from the signal bar
bars_from_signal = df_5m_mapped[df_5m_mapped.index >= test_ts].copy()

# We need at least 10 bars to run a meaningful test
if len(bars_from_signal) < 5:
    print("  ❌ Not enough bars after signal — try an earlier signal bar")
    sys.exit(1)

events = []          # timeline of what happened
trade_open   = False
trade_closed = False
reentry_attempted = False
force_closed_bar  = None  # which bar we forced the close on

print(f"\n  Simulating {len(bars_from_signal)} bars from {test_ts} ...\n")

for i, (ts, row) in enumerate(bars_from_signal.iterrows()):

    bar_signal = int(row["final_signal"]) if not pd.isna(row["final_signal"]) else 0
    ltf_row    = df_1h_sig.iloc[int(row["ltf_index"])]

    # ── FORCE OPEN: on bar 0 we inject the signal regardless ──
    if i == 0:
        bar_signal = test_sig  # ensure entry fires

    # ── FORCE CLOSE: on bar 1 we inject an opposite impulse candle ──
    # We do this by temporarily making the bar look like a massive
    # opposite candle so _opposite_impulse_exit triggers
    sim_row = row.copy()
    if i == 1 and trade_open and not trade_closed:
        direction = test_sig
        atr_val   = float(row["ATR_5M"]) if not pd.isna(row["ATR_5M"]) else float(row["ATR"]) * 0.2
        # Build a candle that is 2× ATR in the opposite direction
        if direction == 1:  # long → bearish impulse
            sim_row["open"]  = row["close"] + atr_val * 2
            sim_row["close"] = row["close"]
            sim_row["high"]  = sim_row["open"]
            sim_row["low"]   = sim_row["close"]
        else:               # short → bullish impulse
            sim_row["open"]  = row["close"] - atr_val * 2
            sim_row["close"] = row["close"]
            sim_row["low"]   = sim_row["open"]
            sim_row["high"]  = sim_row["close"]
        sim_row["volume"] = row.get("volume", 0) * 5  # high volume confirms impulse
        bar_signal = 0   # no new signal on the close bar
        force_closed_bar = ts
        print(f"  [INJECT] Bar {i} @ {ts} — forcing opposite impulse to trigger close")

    # ── After close: re-inject the same signal to test reentry block ──
    if i >= 2 and trade_closed and not reentry_attempted:
        bar_signal = test_sig  # try to re-enter same direction
        print(f"  [INJECT] Bar {i} @ {ts} — re-injecting signal={test_sig} to test reentry block")
        reentry_attempted = True

    result = pm.update(
        df=df_1h_sig,
        symbol=SYMBOL,
        lltf_df=df_5m_mapped,
        external_signal=bar_signal,
        external_row=ltf_row,
        current_5m_row=sim_row,
    )

    state = result.get("state") if isinstance(result, dict) else "FLAT"

    event = {
        "bar"    : i,
        "ts"     : ts,
        "signal" : bar_signal,
        "state"  : state,
        "has_pos": SYMBOL in pm.positions,
        "lock"   : pm._reentry_lock.get(SYMBOL),
        "lock_ts": pm._reentry_lock_ts.get(SYMBOL),
        "executed_count": len(pm._executed_signals),
    }
    events.append(event)

    label = ""
    if state == "OPEN":
        trade_open = True
        label = "  ← TRADE OPENED"
    elif state == "CLOSED":
        trade_closed = True
        label = "  ← TRADE CLOSED"
        if i == 0:
            print("  ⚠️  Trade closed on entry bar — signal may be too weak for this bar")
    elif state == "FLAT" and reentry_attempted and i == 2:
        label = "  ← REENTRY ATTEMPTED (FLAT = blocked ✅  |  OPEN = bug 🔴)"

    print(
        f"  Bar {i:>3} | {ts} | sig={bar_signal:+d} | "
        f"state={state:<6} | pos={'YES' if event['has_pos'] else 'no ':>3} | "
        f"lock={event['lock']} | executed={event['executed_count']}"
        f"{label}"
    )

    # Stop after we've tested the reentry
    if reentry_attempted and i > 2:
        break

# ─────────────────────────────────────────────
# 5. VERDICT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("VERDICT")
print("="*60)

if not trade_open:
    print("  ⚠️  No trade opened — signal may have been filtered or expired")
    print("      Try running again (a different signal bar may be picked)")
elif not trade_closed:
    print("  ⚠️  Trade opened but did not close — opposite impulse injection may")
    print("      not have been strong enough for this bar's ATR. Check bar 1 above.")
elif not reentry_attempted:
    print("  ⚠️  Could not attempt reentry — not enough bars after close")
else:
    # Check what happened on the reentry bar
    reentry_event = next((e for e in events if e["bar"] == 2), None)
    if reentry_event is None:
        print("  ⚠️  Could not find reentry bar in event log")
    elif reentry_event["state"] == "OPEN":
        print("  🔴 FAIL — System RE-ENTERED the same direction immediately after close")
        print("            Reentry lock or executed signal set did NOT block the entry")
        print()
        # Diagnose which guard failed
        lock = reentry_event["lock"]
        lock_ts = reentry_event["lock_ts"]
        if lock is None:
            print("  ► Reentry lock was NOT set — bug is in _close()")
        else:
            print(f"  ► Reentry lock WAS set (dir={lock} locked_at={lock_ts})")
            print(f"  ► But entry still happened — bug is in the unlock timing check")
    elif reentry_event["state"] == "FLAT":
        print("  ✅ PASS — Reentry was correctly blocked after immediate close")
        print(f"            lock={reentry_event['lock']}  executed={reentry_event['executed_count']}")
    else:
        print(f"  ❓ Unexpected state on reentry bar: {reentry_event['state']}")

# ─────────────────────────────────────────────
# 6. DUMP FINAL STATE FILES FOR INSPECTION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STATE FILES (in data/test_reentry/)")
print("="*60)

for fname in os.listdir(TEST_DIR):
    fpath = os.path.join(TEST_DIR, fname)
    try:
        with open(fpath) as f:
            content = json.load(f)
        print(f"\n  {fname}:")
        print("  " + json.dumps(content, indent=4, default=str)[:800])
    except Exception:
        pass

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60 + "\n")