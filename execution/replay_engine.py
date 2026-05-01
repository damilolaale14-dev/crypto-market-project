# execution/replay_engine.py

import os
import json
import pandas as pd
from execution.hourly_runner import run_hourly_for_symbol, SYMBOLS
from strategy.lifecycle import PositionManager
from execution.notifier import TelegramNotifier

def _get_state_files():
    files = [
        "data/last_hour_seen.json",
        "data/positions/open_positions.json",
        "data/positions/bar_history.json",
        "data/positions/executed_signals.json",
        "data/positions/reentry_lock.json",
        "data/positions/last_entry_ts.json",
    ]
    # add all per-symbol cursor files
    if os.path.exists("data/cursors"):
        for f in os.listdir("data/cursors"):
            files.append(os.path.join("data/cursors", f))
    return files

REPLAY_CURSOR_FILE = "data/replay_last_5m_seen.json"

def reset_replay_state(symbols=None):
    files = [
        "data/last_hour_seen.json",
        "data/positions/open_positions.json",
        "data/positions/bar_history.json",
        "data/positions/executed_signals.json",
        "data/positions/reentry_lock.json",
        "data/positions/last_entry_ts.json",
    ]
    for f in files:
        if os.path.exists(f):
            os.remove(f)

    # Only wipe cursor files for targeted symbols
        # Wipe cursor files for targeted symbols
    if os.path.exists("data/cursors"):
        for fname in os.listdir("data/cursors"):
            if symbols and not any(sym in fname for sym in symbols):
                continue
            full_path = os.path.join("data/cursors", fname)
            if os.path.exists(full_path):
                os.remove(full_path)

    # Bust parquet cache for targeted symbols so update_symbol fetches fresh
    if symbols and os.path.exists("data/cache"):
        for fname in os.listdir("data/cache"):
            if any(sym in fname for sym in symbols):
                full_path = os.path.join("data/cache", fname)
                if os.path.exists(full_path):
                    os.remove(full_path)

    # FIX: also wipe live cursor files for targeted symbols so hour memory
    # and 5m cursor stay in sync after a reset
    if os.path.exists("data/cursors"):
        for fname in os.listdir("data/cursors"):
            if fname.startswith("live_"):
                if symbols and not any(sym in fname for sym in symbols):
                    continue
                full_path = os.path.join("data/cursors", fname)
                if os.path.exists(full_path):
                    print(f"[RESET] Wiping live cursor: {fname}")
                    os.remove(full_path)

def fast_replay_symbol(symbol: str, from_ts=None, to_ts=None, notify_trades=True):
    notifier = TelegramNotifier()

    # ==================================================
    # LOAD FULL CACHED DATA (already fetched by fast_replay_all)
    # ==================================================
    df_1h_full   = pd.read_parquet(f"data/cache/{symbol}_1h.parquet")
    df_4h_full   = pd.read_parquet(f"data/cache/{symbol}_4h.parquet")
    df_5m_full   = pd.read_parquet(f"data/cache/{symbol}_5m.parquet")

    df_1h_full.index = pd.to_datetime(df_1h_full.index, utc=True)
    df_4h_full.index = pd.to_datetime(df_4h_full.index, utc=True)
    df_5m_full.index = pd.to_datetime(df_5m_full.index, utc=True)

    # ==================================================
    # APPLY TIME BOUNDS
    # ==================================================
    if to_ts:
        to_ts_parsed = pd.Timestamp(to_ts, tz="UTC")
        df_1h_full = df_1h_full[df_1h_full.index <= to_ts_parsed]
        df_4h_full = df_4h_full[df_4h_full.index <= to_ts_parsed]
        df_5m_full = df_5m_full[df_5m_full.index <= to_ts_parsed]

    if from_ts:
        from_ts_parsed = pd.Timestamp(from_ts, tz="UTC")
    else:
        from_ts_parsed = None

    total_1h = len(df_1h_full)

    # ==================================================
    # SPLIT: WARMUP (first 800) vs ACTIVE (last 200)
    # ==================================================
    WARMUP_BARS = 800

    if total_1h <= WARMUP_BARS:
        notifier.send_text(
            f"⚠️ *REPLAY SKIPPED*\n`{symbol}` — only `{total_1h}` 1H bars, need >{WARMUP_BARS}"
        )
        return

    df_1h_warmup = df_1h_full.iloc[:WARMUP_BARS]
    df_1h_active = df_1h_full.iloc[WARMUP_BARS:]   # last 200

    warmup_end_ts   = df_1h_warmup.index[-1]
    active_start_ts = df_1h_active.index[0]

    notifier.send_text(
        f"🔁 *REPLAY STARTED*\n"
        f"Symbol: `{symbol}`\n"
        f"Total 1H bars: `{total_1h}`\n"
        f"Warmup: `{WARMUP_BARS}` bars → ends `{warmup_end_ts}`\n"
        f"Active: `{len(df_1h_active)}` bars → from `{active_start_ts}`\n"
        f"5m bars total: `{len(df_5m_full)}`"
    )

    # ==================================================
    # PRE-GENERATE SIGNALS ON WARMUP (once, frozen)
    # ==================================================
    from indicators.indicators import generate_signal

    df_warmup_with_signals = generate_signal(df_1h_warmup.copy(), df_4h_full.copy())

    # ==================================================
    # POSITION MANAGER — single instance, in-memory
    # ==================================================
    pm = PositionManager(persist=False, notify=notify_trades)

    trade_opens  = 0
    trade_closes = 0

    # ==================================================
    # INCREMENTAL LOOP OVER ACTIVE 200 1H BARS
    # ==================================================
    for i, (ts_1h, _) in enumerate(df_1h_active.iterrows()):

        # Build growing 1H slice: warmup + active bars seen so far
        # (i=0 → warmup only, i=1 → warmup + first active bar, etc.)
        df_1h_slice = pd.concat([df_1h_warmup, df_1h_active.iloc[:i+1]])
        df_4h_slice  = df_4h_full[df_4h_full.index <= df_1h_slice.index[-1]]

        if len(df_1h_slice) < 2:
            continue

        # Generate signals on the growing slice
        df_signals = generate_signal(df_1h_slice.copy(), df_4h_slice.copy())

        # 5m bars that belong to the current 1H candle
        next_1h_ts = df_1h_active.index[i + 1] if i + 1 < len(df_1h_active) else None
        if next_1h_ts is None:
            break  # no next bar to execute on

        # slice: 5m bars from current 1H open up to (but not including) next 1H open
        mask_5m = (df_5m_full.index >= ts_1h) & (df_5m_full.index < next_1h_ts)
        df_5m_slice = df_5m_full[mask_5m]

        if df_5m_slice.empty:
            continue

        # ── map 5m bars to their parent 1H index ──────────────────
        from execution.hourly_runner import map_ltf_to_htf

        lltf = df_5m_slice.copy()
        lltf = map_ltf_to_htf(lltf, df_signals)

        # forward-fill final_signal from 1H onto 5m
        lltf["final_signal"] = df_signals["final_signal"].reindex(
            lltf.index, method="ffill"
        )

        # precompute 5m ATR
        tr_5m = pd.concat([
            df_5m_full["high"] - df_5m_full["low"],
            (df_5m_full["high"] - df_5m_full["close"].shift()).abs(),
            (df_5m_full["low"]  - df_5m_full["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_5m = tr_5m.rolling(14).mean()
        lltf["ATR"] = atr_5m.reindex(lltf.index)

        # forward-fill 1H ATR onto 5m
        tr_1h = pd.concat([
            df_signals["high"] - df_signals["low"],
            (df_signals["high"] - df_signals["close"].shift()).abs(),
            (df_signals["low"]  - df_signals["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_1h = tr_1h.rolling(14).mean()
        lltf["ATR_1H"] = atr_1h.reindex(lltf.index, method="ffill")

        lltf = lltf.dropna(subset=["ltf_index"])
        lltf["ltf_index"] = lltf["ltf_index"].astype(int)

        if lltf.empty:
            continue

        # skip if before from_ts
        if from_ts_parsed and ts_1h < from_ts_parsed:
            continue

        # ── feed each 5m bar to the position manager ──────────────
        for _, row_5m in lltf.iterrows():
            bar_signal = 0 if pd.isna(row_5m.get("final_signal")) else int(row_5m["final_signal"])
            ltf_row    = df_signals.iloc[int(row_5m["ltf_index"])]

            result = pm.update(
                df=df_signals,
                symbol=symbol,
                lltf_df=lltf,
                external_signal=bar_signal,
                external_row=ltf_row,
                current_5m_row=row_5m,
            )

            if isinstance(result, dict):
                if result.get("state") == "OPEN":
                    trade_opens += 1
                elif result.get("state") == "CLOSED":
                    trade_closes += 1

        if (i + 1) % 24 == 0:
            notifier.send_text(
                f"⏳ *REPLAY PROGRESS*\n"
                f"`{symbol}` — 1H bar {i+1}/{len(df_1h_active)}\n"
                f"ts: `{ts_1h}`\n"
                f"Opens: `{trade_opens}` | Closes: `{trade_closes}`"
            )

    # ==================================================
    # FORCE-CLOSE ANY OPEN POSITION AT END
    # ==================================================
    if symbol in pm.positions:
        last_bar = df_5m_full.iloc[-1]
        closed = pm._close(symbol, float(last_bar["close"]), last_bar.name, "replay_end")
        print(
            f"[REPLAY END CLOSE] {symbol} "
            f"pnl_r={closed['exit']['pnl_r']:.2f} "
            f"bars={closed['bars_in_trade']}"
        )
        trade_closes += 1

    notifier.send_text(
        f"✅ *REPLAY COMPLETE*\n"
        f"Symbol: `{symbol}`\n"
        f"Trades opened: `{trade_opens}`\n"
        f"Trades closed: `{trade_closes}`"
    )

def fast_replay_all(from_ts=None, to_ts=None, notify_trades=True, symbols=None):
    notifier = TelegramNotifier()

    with open("data/replay_lock.json", "w") as f:
        json.dump({"locked": True, "started": pd.Timestamp.now(tz="UTC").isoformat()}, f)

    try:
        reset_replay_state(symbols=symbols)
        target_symbols = symbols if symbols else SYMBOLS

        for symbol in target_symbols:
            try:
                from data_pipeline.updater import update_symbol
                update_symbol(symbol)
            except Exception as e:
                notifier.send_text(f"💥 *FETCH FAILED*\n`{symbol}`\n`{str(e)[:200]}`")
                continue

            fast_replay_symbol(
                symbol, from_ts=from_ts, to_ts=to_ts, notify_trades=notify_trades
            )
    finally:
        if os.path.exists("data/replay_lock.json"):
            os.remove("data/replay_lock.json")

if __name__ == "__main__":
    fast_replay_all()