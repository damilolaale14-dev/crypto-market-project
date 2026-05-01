# execution/hourly_runner.py
import os
import json
from datetime import datetime, timezone
from utils.log import debug, info, trade, error
from utils.logger import log

from data_pipeline.updater import update_symbol
from indicators.indicators import generate_signal
from strategy.lifecycle import PositionManager
from execution.notifier import TelegramNotifier
import pandas as pd

def _tg_debug(msg: str) -> None:
    """Fire-and-forget debug message to Telegram. Never raises."""
    try:
        TelegramNotifier().debug(f"[RUNNER] {msg}")
    except Exception:
        print(f"[TG DEBUG FALLBACK] {msg}")

SYMBOLS = [
    "ETHUSDT", "FILUSDT", "TRXUSDT", "VETUSDT", "UNIUSDT", "DOGEUSDT", "ETCUSDT",
    "AAVEUSDT", "BCHUSDT", "BANDUSDT", "TIAUSDT", "XLMUSDT", "SUIUSDT", "BTCUSDT",
    "ZENUSDT", "AVAXUSDT", "AXSUSDT", "ORDIUSDT", "LDOUSDT", "LINKUSDT"
]

SIGNAL_STORE       = "data/signals.json"
HOUR_MEMORY_FILE = "data/last_hour_seen.json"

# Interval constants
LLTF_INTERVAL = "5m"
LTF_INTERVAL  = "1h"
HTF_INTERVAL  = "4h"

def _last_5m_file(symbol: str, live: bool) -> str:
    prefix = "live" if live else "replay"
    return f"data/cursors/{prefix}_{symbol}.json"

def run_hourly():
    print("\n==============================")
    print("CRYPTO MARKET PROJECT EXECUTION")
    print("==============================\n")

    os.makedirs("data", exist_ok=True)

    notifier = TelegramNotifier()

    if os.path.exists("data/replay_lock.json"):
        notifier.send_text("🔒 *LIVE SKIPPED*\nReplay lock active — skipping live execution")
        return

    with open("data/last_run.json", "w") as f:
        json.dump(
            {
                "ran_at": datetime.now(timezone.utc).isoformat(),
                "symbols": SYMBOLS
            },
            f,
            indent=2
        )

    symbol_summaries = []
    for symbol in SYMBOLS:
        result = run_hourly_for_symbol(symbol)
        if isinstance(result, tuple):
            summary, _ = result
        else:
            summary = result
        symbol_summaries.append((symbol, summary))

    now = datetime.now(timezone.utc)
    local_now = now + pd.Timedelta(hours=1)  # WAT = UTC+1

    # Only send hourly summary — check if we just crossed a new hour
    current_hour = local_now.replace(minute=0, second=0, microsecond=0)
    last_summary_file = "data/last_summary_hour.json"
    last_summary_hour = None
    if os.path.exists(last_summary_file):
        try:
            with open(last_summary_file, "r") as f:
                last_summary_hour = json.load(f).get("hour")
        except Exception:
            pass

    current_hour_str = current_hour.isoformat()
    is_new_hour = last_summary_hour != current_hour_str

    active_lines = []
    for symbol, summary in symbol_summaries:
        if isinstance(summary, list):
            opens  = sum(1 for r in summary if r.get("state") == "OPEN")
            closes = sum(1 for r in summary if r.get("state") == "CLOSED")
            parts = []
            if opens:
                parts.append(f"{opens} opened")
            if closes:
                parts.append(f"{closes} closed")
            if parts:
                active_lines.append(f"`{symbol}` — " + ", ".join(parts))

    # Always notify if there are trades
    has_trades = bool(active_lines)

    if is_new_hour or has_trades:
        ran_at = local_now.strftime("%H:%M WAT")
        actual_ran_at = local_now.strftime("%H:%M:%S WAT")
        msg = f"🕐 *LIVE RUN* `{ran_at}` | triggered `{actual_ran_at}`"
        if active_lines:
            msg += "\n" + "\n".join(active_lines)

        notifier.send_text(msg)

        if is_new_hour:
            with open(last_summary_file + ".tmp", "w") as f:
                json.dump({"hour": current_hour_str}, f)
            os.replace(last_summary_file + ".tmp", last_summary_file)

    print("\n=== EXECUTION COMPLETE ===\n")

# ==========================================================
# SINGLE SYMBOL ENGINE (UNIFIED LIVE + REPLAY)
# ==========================================================
def run_hourly_for_symbol(
    symbol: str,
    forced_time=None,
    replay=False,
    notify_override=None,
    verbose=True,
    replay_cursor=None,
    external_pm=None,          # FIX 2: accept shared PM from replay caller
):
    is_live = not replay and forced_time is None
    notify = notify_override if notify_override is not None else is_live
    notifier = TelegramNotifier()

    # -------------------
    # FAST GATE — skip entire symbol if no new 5m bar (LIVE ONLY)
    # -------------------
    if is_live:
        cursor_file = _last_5m_file(symbol, True)
        if os.path.exists(cursor_file):
            try:
                with open(cursor_file, "r") as f:
                    raw = json.load(f)
                raw_val = raw if isinstance(raw, str) else list(raw.values())[0]
                last_seen_ts = pd.Timestamp(raw_val)
                if last_seen_ts.tzinfo is None:
                    last_seen_ts = last_seen_ts.tz_localize("UTC")
                else:
                    last_seen_ts = last_seen_ts.tz_convert("UTC")
                now_check = datetime.now(timezone.utc)
                minutes_floored = (now_check.minute // 5) * 5
                current_5m_boundary = now_check.replace(minute=minutes_floored, second=0, microsecond=0)

                if last_seen_ts > pd.Timestamp(now_check).tz_convert("UTC") + pd.Timedelta(hours=1):
                    _tg_debug(f"[FAST GATE POISONED] {symbol} — cursor {last_seen_ts} is in the future, deleting and proceeding")
                    notifier.send_text(
                        f"⚠️ *CURSOR POISONED*\n"
                        f"Symbol: `{symbol}`\n"
                        f"Cursor: `{last_seen_ts}`\n"
                        f"Now: `{now_check}`\n"
                        f"Deleting and reprocessing last 12 bars"
                    )
                    os.remove(cursor_file)
                elif last_seen_ts >= pd.Timestamp(current_5m_boundary).tz_convert("UTC"):
                    print(f"[FAST GATE] {symbol} — cursor {last_seen_ts} >= boundary {current_5m_boundary}, skipping")
                    return None
                else:
                    print(f"[FAST GATE PASS] {symbol} — cursor {last_seen_ts} < boundary {current_5m_boundary}, proceeding")

            except Exception as e:
                _tg_debug(f"[FAST GATE ERROR] {symbol} — {e}, proceeding")

    # FIX 2: use external PM if provided (replay), else instantiate normally
    if external_pm is not None:
        pm = external_pm
    else:
        pm = PositionManager(persist=True, notify=notify)

    # =========================
    # 5M STREAM MEMORY
    # =========================
    os.makedirs("data/cursors", exist_ok=True)
    last_5m_file = _last_5m_file(symbol, is_live)
    try:
        if os.path.exists(last_5m_file):
            with open(last_5m_file, "r") as f:
                last_seen_raw = json.load(f)
                last_5m_seen = {symbol: last_seen_raw} if isinstance(last_seen_raw, str) else last_seen_raw
        else:
            last_5m_seen = {}
    except Exception as state_err:
        notifier.send_text(
            f"💥 *STATE LOAD FAILED*\n"
            f"`{symbol}`\n"
            f"file=`{last_5m_file}`\n"
            f"error=`{str(state_err)[:200]}`"
        )
        last_5m_seen = {}

    try:
        # -------------------
        # FETCH DATA
        # -------------------
        try:
            if forced_time is None and not replay:
                df, htf_df, lltf_df = update_symbol(symbol)
            else:
                df, htf_df, lltf_df = update_symbol(symbol)

                if forced_time:
                    df      = df[df.index < forced_time].copy()
                    htf_df  = htf_df[htf_df.index < forced_time].copy()
                    lltf_df = lltf_df[lltf_df.index < forced_time].copy()

                    if len(df) < 2 or len(htf_df) < 2 or len(lltf_df) < 2:
                        _tg_debug(f"[WARMUP SKIP] {symbol} forced_time={forced_time} — insufficient data (1h={len(df)} 4h={len(htf_df)} 5m={len(lltf_df)})")
                        return None, replay_cursor
                else:
                    df, htf_df, lltf_df = df.iloc[:-1], htf_df.iloc[:-1], lltf_df.iloc[:-1]

        except Exception as fetch_err:
            notifier.send_text(
                f"💥 *UPDATE_SYMBOL FAILED*\n"
                f"`{symbol}` forced_time=`{forced_time}`\n"
                f"Error: `{str(fetch_err)[:300]}`"
            )
            return None

        # -------------------
        # GENERATE & MAP SIGNALS
        # -------------------
        df = generate_signal(df.copy(), htf_df.copy())

        lltf_df = lltf_df[lltf_df.index >= df.index[0]].copy()
        lltf_df = map_ltf_to_htf(lltf_df, df)

        lltf_df["final_signal"] = df["final_signal"].reindex(
            lltf_df.index,
            method="ffill"
        )

        if 'final_signal' not in df.columns or len(df) < 2:
            return

        # FIX 5: diagnostic — log signal state so we can see if signals are reaching this point
        non_null_signals = lltf_df["final_signal"].notna().sum()
        non_zero_signals = (lltf_df["final_signal"] != 0).sum()
        # _tg_debug(f"[SIGNAL DIAG] {symbol} — non-null={non_null_signals} non-zero={non_zero_signals} total_5m_bars={len(lltf_df)}")

        # Precompute rolling ATR on 5m dataframe
        tr_5m = pd.concat([
            lltf_df['high'] - lltf_df['low'],
            (lltf_df['high'] - lltf_df['close'].shift()).abs(),
            (lltf_df['low']  - lltf_df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        lltf_df['ATR'] = tr_5m.rolling(14).mean()

        lltf_frozen = lltf_df.copy()
        lltf_frozen = lltf_frozen.dropna(subset=['ltf_index'])
        lltf_frozen['ltf_index'] = lltf_frozen['ltf_index'].astype(int)

        # FIX 5: diagnostic — log how many bars survive dropna
        print(f"[FROZEN DIAG] {symbol} — bars after dropna={len(lltf_frozen)}")

        # ==========================================================
        # NEW 1H CANDLE DETECTION
        # ==========================================================
        if os.path.exists(HOUR_MEMORY_FILE):
            with open(HOUR_MEMORY_FILE, "r") as f:
                last_hour_seen = json.load(f)
        else:
            last_hour_seen = {}

        latest_hour_ts = df.index[-1].isoformat()
        previous_hour  = last_hour_seen.get(symbol)
        new_hour = latest_hour_ts != previous_hour

        # =========================
        # STREAMING ENGINE
        # =========================
        latest_ts = lltf_frozen.index[-1]
        if replay_cursor is not None:
            last_seen = replay_cursor
        else:
            raw = last_5m_seen.get(symbol) or (last_5m_seen if isinstance(last_5m_seen, str) else None)
            last_seen = pd.Timestamp(raw) if raw else None

        if is_live and last_seen == latest_ts:
            return None

        if last_seen is None and not replay and not forced_time:
            notifier.send_text(
                f"⚠️ *CURSOR RESET DETECTED*\n"
                f"Symbol: `{symbol}`\n"
                f"Processing last 12 bars to recover signal state"
            )
            recovery_bars = 12
            if len(lltf_frozen) > recovery_bars:
                last_seen = lltf_frozen.index[-(recovery_bars + 1)]
            else:
                last_seen = lltf_frozen.index[0]

        new_bars = (
            lltf_frozen if last_seen is None
            else lltf_frozen[lltf_frozen.index > last_seen]
        )

        # FIX 5: diagnostic — log new_bars count so we know if streaming engine sees anything
        print(f"[NEW BARS DIAG] {symbol} — new_bars={len(new_bars)} last_seen={last_seen} latest_ts={latest_ts}")

        if new_bars.empty:
            return None

        bar_results = []

        for _, row_5m in new_bars.iterrows():

            if pd.isna(row_5m["final_signal"]):
                bar_signal = 0
            else:
                bar_signal = int(row_5m["final_signal"])

            ltf_row = df.iloc[int(row_5m["ltf_index"])]

            if not pd.isna(row_5m.get("final_signal", float("nan"))) and row_5m["final_signal"] != 0:
                notifier.send_text(
                    f"🚨 *SIGNAL REACHED LIFECYCLE*\n"
                    f"{symbol}\n"
                    f"ts: `{_}`\n"
                    f"signal: `{row_5m['final_signal']}`"
                )

            # FIX 5: diagnostic — log what bar_signal is actually passed to pm.update
            # _tg_debug(f"[PM UPDATE DIAG] {symbol} ts={_} bar_signal={bar_signal} has_position={symbol in pm.positions}")

            result = pm.update(
                df=df,
                symbol=symbol,
                lltf_df=lltf_frozen,
                external_signal=bar_signal,
                external_row=ltf_row,
                current_5m_row=row_5m
            )
            if isinstance(result, dict) and result.get("state") in ("OPEN", "CLOSED"):
                bar_results.append(result)

        # FIX 1: replay-end close REMOVED from here — caller (fast_replay_symbol) owns it
        # This was the bug causing open+close on every single bar

        # update cursor AFTER processing (live only)
        if not replay and replay_cursor is None:
            with open(last_5m_file + ".tmp", "w") as f:
                json.dump(new_bars.index[-1].isoformat(), f)
            os.replace(last_5m_file + ".tmp", last_5m_file)

        # ==========================================================
        # SAVE LAST PROCESSED HOUR
        # ==========================================================
        if not replay and not forced_time:
            cursor_file = _last_5m_file(symbol, True)
            cursor_exists = os.path.exists(cursor_file)

            # FIX: if cursor was just reset (file didn't exist before this run),
            # force hour memory to reset too so the two clocks stay in sync
            if not cursor_exists and not new_hour:
                _tg_debug(f"[CLOCK SYNC] {symbol} — cursor was absent but hour memory has entry, forcing hour reset")
                new_hour = True

            if new_hour:
                last_hour_seen[symbol] = latest_hour_ts
                with open(HOUR_MEMORY_FILE + ".tmp", "w") as f:
                    json.dump(last_hour_seen, f, indent=2)
                os.replace(HOUR_MEMORY_FILE + ".tmp", HOUR_MEMORY_FILE)
                print(f"[HOUR MEMORY UPDATED] {symbol} — {latest_hour_ts}")
            else:
                print(f"[HOUR MEMORY UNCHANGED] {symbol} — already at {latest_hour_ts}")

        pm.flush()

        new_cursor = new_bars.index[-1] if not new_bars.empty else replay_cursor
        return (bar_results if bar_results else None), new_cursor

    except Exception as e:
        import traceback
        notifier.send_text(
            f"💥 *RUNNER EXCEPTION*\n"
            f"`{symbol}` forced_time=`{forced_time}`\n"
            f"error=`{str(e)[:300]}`"
        )
        error(f"[ERROR] {symbol} → {e}")
        traceback.print_exc()
        return None

def map_ltf_to_htf(lltf_df: pd.DataFrame, htf_df: pd.DataFrame):

    htf_times = htf_df.index

    ltf_index = []

    for ts in lltf_df.index:

        # find correct 1H candle start
        idx = htf_times.searchsorted(ts, side="right") - 1

        if idx < 0:
            idx = 0

        ltf_index.append(idx)

    lltf_df = lltf_df.copy()
    lltf_df["ltf_index"] = ltf_index

    return lltf_df