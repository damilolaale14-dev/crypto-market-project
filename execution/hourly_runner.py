# execution/hourly_runner.py
import os
import json
from datetime import datetime, timezone
from utils.log import debug, info, trade, error
from utils.logger import log

from data_pipeline.updater import update_symbol
from indicators.indicators import generate_signal, atr_ema
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
    "AXSUSDT", "XRPUSDT", "AVAXUSDT", "DOTUSDT", "AAVEUSDT", "XLMUSDT", 
    "SUIUSDT", "VETUSDT", "TRXUSDT", "LDOUSDT", "INJUSDT", "RUNEUSDT", 
    "ORDIUSDT", "ADAUSDT", "ZENUSDT", "TIAUSDT", "OPUSDT", "ICPUSDT", 
    "PAXGUSDT", "TRBUSDT"
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
    failed_symbols = []
    for symbol in SYMBOLS:
        try:
            result = run_hourly_for_symbol(symbol)
            if isinstance(result, tuple):
                summary, _ = result
            else:
                summary = result
            symbol_summaries.append((symbol, summary))
        except Exception as sym_err:
            import traceback
            tb = traceback.format_exc()
            failed_symbols.append(symbol)
            notifier.send_text(
                f"💥 *SYMBOL CRASH*\n"
                f"Symbol: `{symbol}`\n"
                f"Error: `{str(sym_err)[:300]}`\n"
                f"Traceback:\n`{tb[:600]}`"
            )
            symbol_summaries.append((symbol, None))

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
                current_5m_boundary = now_check.replace(
                    minute=minutes_floored, second=0, microsecond=0
                )
                current_5m_boundary = pd.Timestamp(current_5m_boundary).tz_convert("UTC")

                if last_seen_ts > pd.Timestamp(now_check).tz_convert("UTC") + pd.Timedelta(hours=1):
                    _tg_debug(
                        f"[FAST GATE POISONED] {symbol} — cursor {last_seen_ts} "
                        f"is in the future, deleting and proceeding"
                    )
                    notifier.send_text(
                        f"⚠️ *CURSOR POISONED*\n"
                        f"Symbol: `{symbol}`\n"
                        f"Cursor: `{last_seen_ts}`\n"
                        f"Now: `{now_check}`\n"
                        f"Deleting and reprocessing last 12 bars"
                    )
                    os.remove(cursor_file)
                elif last_seen_ts >= current_5m_boundary:
                    # Cursor is current — but only skip if we also have no open position.
                    # If a position is open we must keep running exit checks every bar.
                    pm_check = PositionManager(persist=True, notify=False)
                    if symbol not in pm_check.positions:
                        print(
                            f"[FAST GATE] {symbol} — cursor {last_seen_ts} >= "
                            f"boundary {current_5m_boundary}, no open position, skipping"
                        )
                        return None
                    else:
                        print(
                            f"[FAST GATE BYPASS] {symbol} — cursor current but "
                            f"position open, proceeding for exit checks"
                        )
                else:
                    print(
                        f"[FAST GATE PASS] {symbol} — cursor {last_seen_ts} < "
                        f"boundary {current_5m_boundary}, proceeding"
                    )

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
                df, htf_df, lltf_df, htf_scores = update_symbol(symbol)
            else:
                df, htf_df, lltf_df, htf_scores = update_symbol(symbol)

                if forced_time:
                    df      = df[df.index <= forced_time].copy()
                    htf_df  = htf_df[htf_df.index <= forced_time].copy()
                    lltf_df = lltf_df[lltf_df.index < forced_time].copy()
                    # trim scores to forced_time as well
                    if htf_scores is not None:
                        htf_scores = htf_scores[htf_scores.index <= forced_time].copy()

                    if len(df) < 2 or len(htf_df) < 2 or len(lltf_df) < 2:
                        _tg_debug(f"[WARMUP SKIP] {symbol} forced_time={forced_time} — insufficient data (1h={len(df)} 4h={len(htf_df)} 5m={len(lltf_df)})")
                        return None, replay_cursor
                else:
                    pass

        except Exception as fetch_err:
            notifier.send_text(
                f"💥 *UPDATE_SYMBOL FAILED*\n"
                f"`{symbol}` forced_time=`{forced_time}`\n"
                f"Error: `{str(fetch_err)[:300]}`"
            )
            return None

        # -------------------
        # INCOMPLETE CANDLE GUARD (live only — replay/backtest unaffected)
        # -------------------
        now_utc = datetime.now(timezone.utc)
        minutes_floored = (now_utc.minute // 5) * 5
        current_5m_boundary = pd.Timestamp(
            now_utc.replace(minute=minutes_floored, second=0, microsecond=0)
        ).tz_convert("UTC")

        now_utc_ts = pd.Timestamp(now_utc).tz_convert("UTC")
        seconds_elapsed = (now_utc_ts - current_5m_boundary).total_seconds()
        boundary_in_data = current_5m_boundary in lltf_df.index
        boundary_is_hour_open = current_5m_boundary.minute == 0

        # generate_signal hasn't run yet here — check after it runs below
        # (split the guard: include decision is made after signal gen)
        _early_entry_eligible = (
            is_live
            and seconds_elapsed >= 30
            and boundary_in_data
            and boundary_is_hour_open
        )

        if _early_entry_eligible:
            lltf_df = lltf_df[lltf_df.index <= current_5m_boundary].copy()
            print(f"[EARLY ENTRY GUARD] {symbol} — boundary bar {current_5m_boundary} included ({seconds_elapsed:.0f}s elapsed)")
        else:
            lltf_df = lltf_df[lltf_df.index < current_5m_boundary].copy()

        # notifier.debug(
        #     f"[CANDLE GUARD] {symbol}\n"
        #     f"5m_boundary={current_5m_boundary}\n"
        #     f"lltf_last_before={lltf_last_before}\n"
        #     f"lltf_last_after={lltf_df.index[-1] if not lltf_df.empty else 'EMPTY'}\n"
        #     f"incomplete_bar_removed={lltf_last_before >= current_5m_boundary}"
        # )

        if lltf_df.empty:
            notifier.debug(f"[CANDLE GUARD EMPTY] {symbol} — lltf_df empty after clip, skipping")
            return None, replay_cursor

        # -------------------
        # GENERATE & MAP SIGNALS
        # -------------------
        df = generate_signal(df.copy(), htf_df.copy(), live=is_live, symbol=symbol, htf_stack_cache=htf_scores)

        _htf_quality   = float(df['HTF_QUALITY'].iloc[-1])
        _htf_direction = int(df['HTF_DIRECTION'].iloc[-1])
        _signal_count  = int((df['final_signal'] != 0).sum())

        print(
            f"[HTF CHECK] {symbol} | "
            f"HTF_QUALITY={_htf_quality:.4f} | "
            f"HTF_DIRECTION={_htf_direction} | "
            f"final_signal_last={df['final_signal'].iloc[-1]} | "
            f"signals_total={_signal_count}"
        )

        if 'final_signal' not in df.columns or len(df) < 2:
            notifier.debug(f"[SIGNAL GUARD] {symbol} — no final_signal or df too short")
            return None, replay_cursor

        lltf_df = lltf_df[lltf_df.index >= df.index[0]].copy()
        lltf_df = map_ltf_to_htf(lltf_df, df)

        lltf_df["final_signal"] = df["final_signal"].reindex(
            lltf_df.index,
            method="ffill"
        )

        # Zero out 5m bars that are INSIDE the 1H bar that generated their signal.
        # A signal on the 12:00 1H bar is only valid starting at 13:00 (when that bar closes).
        # 5m bars at 12:05, 12:10, ..., 12:55 must be zeroed.
        # The 13:00 bar maps to the NEW 1H bar (13:00), so it is NOT zeroed here.
        #
        # Rule: zero a 5m bar if its originating 1H bar open == its own timestamp floored to 1H.
        # Equivalently: zero if ts < (origin + 1H) AND ts >= origin AND origin == floor(ts, 1H).
        # Simplified: zero if the 5m bar's timestamp falls strictly BEFORE its 1H bar's close.
        # A 5m bar AT the 1H boundary (e.g. 13:00) already maps to the NEW 1H bar via
        # map_ltf_to_htf (searchsorted right - 1 gives the new bar), so it is safe.

        if 'final_signal' in lltf_df.columns:
            # Floor each 5m timestamp to its 1H bar open
            ts_1h_floor = lltf_df.index.floor('h')
            # Get the 1H bar open each 5m bar maps to via ltf_index
            ltf_opens = df.index
            mapped_1h_open = lltf_df['ltf_index'].apply(
                lambda i: ltf_opens[int(i)] if 0 <= int(i) < len(ltf_opens) else pd.NaT
            )
            # A 5m bar is INSIDE its generating 1H bar when its timestamp floor equals
            # its mapped 1H open — meaning the 1H bar hasn't closed yet at this 5m bar.
            # (The 13:00 5m bar maps to the 13:00 1H bar, floor is 13:00, equals — so
            #  it WOULD be blocked. But 13:00 is a valid entry. We want to allow it.)
            # 
            # Correct rule: block if ts is STRICTLY BETWEEN origin and origin+1H (exclusive both ends
            # means we allow the boundary). Actually: block if origin <= ts < origin+1H
            # but ts != origin (don't block the open of the new bar).
            # Since the new 1H bar opens AT origin, and that IS a valid entry bar,
            # we block: origin < ts < origin+1H  (strictly inside, not at boundary).
            #
            # But ts AT origin: that's the very first 5m bar of the new hour. That bar's
            # ltf_index maps to the 1H bar that just closed (index N-1), not the new one,
            # because map_ltf_to_htf uses searchsorted('right') - 1. So origin for that
            # bar = the PREVIOUS 1H bar open. So it WILL be blocked correctly.
            # The 5m bar at 13:00 maps to 1H bar at 12:00 (last closed bar). origin=12:00.
            # 12:00 <= 13:00 < 13:00 → False (13:00 < 13:00 is False). NOT blocked. ✓
            
            block_mask = pd.Series(False, index=lltf_df.index)
            for ts, ltf_idx in zip(lltf_df.index, lltf_df['ltf_index']):
                idx = int(ltf_idx)
                if idx < 0 or idx >= len(ltf_opens):
                    continue
                origin = ltf_opens[idx]
                origin_end = origin + pd.Timedelta(hours=1)
                if origin <= ts < origin_end:
                    block_mask[ts] = True
            
            lltf_df.loc[block_mask, 'final_signal'] = 0

        # notifier.debug(
        #     f"[SIGNAL GEN] {symbol}\n"
        #     f"df_last={df.index[-1]}\n"
        #     f"non_zero_signals={(df['signal_live'] != 0).sum()}\n"
        #     f"last_15={df['signal_live'].iloc[-15:].tolist()}"
        # )

        # notifier.debug(
        #     f"[SIGNAL MAP] {symbol}\n"
        #     f"lltf_len={len(lltf_df)}\n"
        #     f"lltf_last={lltf_df.index[-1]}\n"
        #     f"non_zero_5m={(lltf_df['final_signal'] != 0).sum()}\n"
        #     f"value_counts={lltf_df['final_signal'].value_counts().to_dict()}"
        # )

        # Use 1H ATR to match backtest — forward fill onto 5m bars
        lltf_df['ATR'] = df['ATR'].reindex(lltf_df.index, method='ffill')
        lltf_df['ATR_5M'] = atr_ema(lltf_df, period=14)

        lltf_frozen = lltf_df.copy()
        lltf_frozen = lltf_frozen.dropna(subset=['ltf_index'])
        lltf_frozen['ltf_index'] = lltf_frozen['ltf_index'].astype(int)

        # notifier.debug(
        #     f"[FROZEN] {symbol}\n"
        #     f"bars_after_dropna={len(lltf_frozen)}\n"
        #     f"frozen_last={lltf_frozen.index[-1] if not lltf_frozen.empty else 'EMPTY'}\n"
        #     f"non_zero_frozen={(lltf_frozen['final_signal'] != 0).sum()}"
        # )

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

        # Alert when HTF filter is blocking all signals.
        # Without this, zero signals from a blocked HTF filter is
        # indistinguishable from zero signals from no setups.
        # new_hour guard fires once per hour per symbol, not every cron run.
        # if _htf_quality <= 0.45 and new_hour:
        #     notifier.debug(
        #         f"[HTF BLOCKED] {symbol} | "
        #         f"quality={_htf_quality:.4f} threshold=0.45 | "
        #         f"dir={_htf_direction} | "
        #         f"no signals will fire this hour"
        #     )

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
            # notifier.debug(f"[CURSOR AT TIP] {symbol} — last_seen={last_seen} == latest_ts={latest_ts}, nothing to do")
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

        print(
            f"[NEW BARS] {symbol} | count={len(new_bars)} | "
            f"non_zero={(new_bars['final_signal'] != 0).sum() if not new_bars.empty else 0} | "
            f"last_seen={last_seen} | latest_ts={latest_ts}"
        )

        # notifier.debug(
        #     f"[NEW BARS] {symbol}\n"
        #     f"new_bars={len(new_bars)}\n"
        #     f"last_seen={last_seen}\n"
        #     f"latest_ts={latest_ts}\n"
        #     f"new_bars_first={new_bars.index[0] if not new_bars.empty else 'EMPTY'}\n"
        #     f"new_bars_last={new_bars.index[-1] if not new_bars.empty else 'EMPTY'}\n"
        #     f"non_zero_in_new_bars={(new_bars['final_signal'] != 0).sum() if not new_bars.empty else 0}"
        # )

        if new_bars.empty:
            notifier.debug(
                f"[EMPTY NEW BARS] {symbol} — no new bars to process\n"
                f"last_seen={last_seen}\n"
                f"latest_ts={latest_ts}\n"
                f"lltf_frozen_range={lltf_frozen.index[0]} → {lltf_frozen.index[-1]}"
            )
            return None

        bar_results = []

        # notifier.debug(f"[REPLAY LOOP] {symbol} — processing {len(new_bars)} bars from {new_bars.index[0]} to {new_bars.index[-1]}")
        _current_signal_birth = None  # anchors signal expiry to first signal bar
        for _, row_5m in new_bars.iterrows():

            if pd.isna(row_5m["final_signal"]):
                bar_signal = 0
            else:
                bar_signal = int(row_5m["final_signal"])

            ltf_row = df.iloc[int(row_5m["ltf_index"])]

            # Anchor expiry to signal birth — the first 1H bar where this
            # signal appeared. _current_signal_birth does not advance with
            # each new 1H bar, so expiry is measured correctly.
            # Resets to None when signal goes flat so the next signal
            # gets its own fresh birth anchor.
            if bar_signal != 0:
                if _current_signal_birth is None:
                    _current_signal_birth = ltf_row
                signal_birth_row = _current_signal_birth
            else:
                _current_signal_birth = None
                signal_birth_row = ltf_row

            if bar_signal != 0:
                print(
                    f"[SIGNAL BAR] {symbol} | ts={_} | signal={bar_signal} | "
                    f"ltf_index={int(row_5m['ltf_index'])} | "
                    f"reentry_lock={pm._reentry_lock.get(symbol)} | "
                    f"has_position={symbol in pm.positions} | "
                    f"executed_count={len(pm._executed_signals)}"
                )

            result = pm.update(
                df=df,
                symbol=symbol,
                lltf_df=lltf_frozen,
                external_signal=bar_signal,
                external_row=signal_birth_row,
                current_5m_row=row_5m
            )

            if bar_signal != 0:
                print(
                    f"[PM RESULT] {symbol} | ts={_} | "
                    f"state={result.get('state') if isinstance(result, dict) else result}"
                )
                
            if isinstance(result, dict) and result.get("state") in ("OPEN", "CLOSED"):
                bar_results.append(result)
                if result.get("state") == "OPEN" and bar_signal != 0:
                    notifier.debug(
                        f"🚨 SIGNAL ENTERED | {symbol} | ts={_} | "
                        f"signal={bar_signal} | "
                        f"1H_ts={ltf_row.name} | "
                        f"1H_open={ltf_row['open']:.6f} | "
                        f"df_last={df.index[-1]} | "
                        f"ltf_index={int(row_5m['ltf_index'])}"
                    )

            has_position = symbol in pm.positions
            if has_position:
                pos = pm.positions[symbol]
                notifier.debug(
                    f"📊 TRADE ACTIVE | {symbol} | ts={TelegramNotifier._fmt_ts(_)} | "
                    f"side={'LONG' if pos['direction'] == 1 else 'SHORT'} | "
                    f"entry={pos['entry_price']:.6f} | "
                    f"stop={pos['stop_loss']:.6f} | "
                    f"bars={pos.get('bars_in_trade', 0)} | "
                    f"pnl={pos.get('pnl_r', 0.0):+.3f}R"
                )

        if not replay and replay_cursor is None:
            if not new_bars.empty:
                current_1h_open = pd.Timestamp(datetime.now(timezone.utc)).floor("h")
                has_open_position = symbol in pm.positions

                if has_open_position:
                    last_clean_ts = new_bars.index[-1]
                else:
                    # advance cursor fully — candle guard in update_symbol
                    # already prevents incomplete bars from leaking in
                    last_clean_ts = new_bars.index[-1]

                if last_clean_ts is not None:
                    with open(last_5m_file + ".tmp", "w") as f:
                        json.dump(last_clean_ts.isoformat(), f)
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

        # ============================================================
        # DEBUG — dump full state after every symbol run
        # ============================================================
        try:
            import json as _json
            _reentry = {k: {"direction": v, "locked_at": str(pm._reentry_lock_ts.get(k))} for k, v in pm._reentry_lock.items()}
            _executed = [s for s in pm._executed_signals if symbol in s]
            _positions = {k: {"direction": v.get("direction"), "bars": v.get("bars_in_trade"), "entry": v.get("entry_time")} for k, v in pm.positions.items()}
            # notifier.debug(
            #     f"[STATE DUMP] {symbol}\n"
            #     f"cursor_saved={new_bars.index[-1].isoformat() if not new_bars.empty else 'unchanged'}\n"
            #     f"positions={_positions}\n"
            #     f"reentry_lock={_reentry}\n"
            #     f"executed_signals_for_symbol={_executed}"
            # )
        except Exception as _e:
            notifier.debug(f"[STATE DUMP FAILED] {symbol} — {_e}")
        # ============================================================

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

def map_ltf_to_htf(lltf_df: pd.DataFrame, ltf_df: pd.DataFrame):
    """
    Maps each 5m bar to its parent 1H candle index.
    ltf_df MUST be the 1H dataframe.
    Passing 4H data silently corrupts every ltf_index, ATR, and entry price.
    This assertion catches that immediately instead of trading on wrong data.
    """
    if len(ltf_df) >= 3:
        _inferred = pd.infer_freq(ltf_df.index[:min(10, len(ltf_df))])
        if _inferred not in (None, "h", "1h", "H", "1H", "60min", "60T", "T60"):
            raise ValueError(
                f"map_ltf_to_htf: expected 1H dataframe, "
                f"got inferred freq='{_inferred}'. "
                f"Pass the 1H df, not the 4H df."
            )

    ltf_times = ltf_df.index

    ltf_index = []

    for ts in lltf_df.index:

        # find correct 1H candle start
        idx = ltf_times.searchsorted(ts, side="right") - 1

        if idx < 0:
            idx = 0

        ltf_index.append(idx)

    lltf_df = lltf_df.copy()
    lltf_df["ltf_index"] = ltf_index

    return lltf_df