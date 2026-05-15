from flask import Flask, request, abort
import os
import json
import threading

from execution.notifier import TelegramNotifier
from execution.hourly_runner import run_hourly, SYMBOLS

app = Flask(__name__)
_run_lock = threading.Lock()

# ==================================================
# CORE ENDPOINTS
# ==================================================

@app.route("/health")
def health():
    return {"status": "alive"}, 200


@app.route("/")
def run():
    if not _run_lock.acquire(blocking=False):
        print("[RUN] Already running — skipping duplicate trigger")
        return {"status": "already_running"}, 200

    def run_and_release():
        try:
            run_hourly()
        finally:
            _run_lock.release()

    thread = threading.Thread(target=run_and_release)
    thread.daemon = True
    thread.start()

    return {"status": "started"}, 200


@app.route("/test-telegram")
def test_telegram():
    notifier = TelegramNotifier()
    notifier.send_text("✅ Telegram connected to Render successfully")
    return {"status": "telegram_test_sent"}, 200


# ==================================================
# DEBUG / OBSERVABILITY ENDPOINTS
# ==================================================

@app.route("/debug/env")
def debug_env():
    return {
        "BOT_TOKEN_SET": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "CHAT_ID_SET": bool(os.getenv("TELEGRAM_CHAT_ID")),
        "RUN_KEY_SET": bool(os.getenv("RUN_KEY")),
    }

@app.route("/debug/signals")
def debug_signals():

    path = "data/signals.json"

    if not os.path.exists(path):
        return {"exists": False}

    with open(path) as f:
        return {"exists": True, "signals": json.load(f)}


@app.route("/debug/run")
def debug_last_run():
    """
    GUARANTEE #1 — proves /run executed
    """
    path = "data/last_run.json"
    if not os.path.exists(path):
        return {"exists": False}

    with open(path, "r") as f:
        return {"exists": True, "run": json.load(f)}
    
@app.route("/test-pipeline")
def test_pipeline():
    if request.args.get("key") != os.getenv("RUN_KEY", "local"):
        abort(403)

    import threading

    def run_test():
        from execution.notifier import TelegramNotifier
        from data_pipeline.fetcher import fetch_ohlcv
        from datetime import datetime, timezone, timedelta
        import pandas as pd
        import os

        notifier = TelegramNotifier()
        symbol = "LDOUSDT"
        os.makedirs("data/cache", exist_ok=True)

        notifier.send_text(
            f"🧪 *PIPELINE TEST STARTED*\n"
            f"Symbol: `{symbol}`\n"
            f"Plan: download 800×1h warmup, then simulate 200 hourly cron ticks"
        )

        try:
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

            # ── PHASE 1: Download warmup ───────────────────────────────
            warmup_end   = now - timedelta(hours=200)
            warmup_start = warmup_end - timedelta(hours=800)

            notifier.send_text(
                f"📥 *PHASE 1: Downloading warmup*\n"
                f"from `{warmup_start}` to `{warmup_end}`"
            )

            df_1h = fetch_ohlcv(symbol, start=warmup_start, end=warmup_end, interval="1h", limit=1000, verbose=False)
            df_4h = fetch_ohlcv(symbol, start=warmup_start, end=warmup_end, interval="4h", limit=1000, verbose=False)
            df_5m = fetch_ohlcv(symbol, start=warmup_start, end=warmup_end, interval="5m", limit=1000, verbose=False)

            df_1h.to_parquet(f"data/cache/{symbol}_1h.parquet")
            df_4h.to_parquet(f"data/cache/{symbol}_4h.parquet")
            df_5m.to_parquet(f"data/cache/{symbol}_5m.parquet")

            notifier.send_text(
                f"💾 *WARMUP SAVED*\n"
                f"1h=`{len(df_1h)}` 4h=`{len(df_4h)}` 5m=`{len(df_5m)}`\n"
                f"Cursor at `{warmup_end}` — starting sim loop"
            )

            # ── PHASE 2: Simulated hourly cron loop ───────────────────
            fake_now = warmup_end

            for tick in range(1, 201):
                fake_now += timedelta(hours=1)

                # reload current parquets
                base_1h = pd.read_parquet(f"data/cache/{symbol}_1h.parquet")
                base_4h = pd.read_parquet(f"data/cache/{symbol}_4h.parquet")
                base_5m = pd.read_parquet(f"data/cache/{symbol}_5m.parquet")

                for df in (base_1h, base_4h, base_5m):
                    df.index = pd.to_datetime(df.index, utc=True)

                cursor_1h = base_1h.index[-1]
                cursor_4h = base_4h.index[-1]
                cursor_5m = base_5m.index[-1]

                # fetch only what's new since last cursor
                new_1h = fetch_ohlcv(symbol, start=cursor_1h, end=fake_now, interval="1h", limit=100, verbose=False)
                new_4h = fetch_ohlcv(symbol, start=cursor_4h, end=fake_now, interval="4h", limit=100, verbose=False)
                new_5m = fetch_ohlcv(symbol, start=cursor_5m, end=fake_now, interval="5m", limit=100, verbose=False)

                # strip rows already in base (cursor row itself may be returned)
                new_1h = new_1h[new_1h.index > cursor_1h]
                new_4h = new_4h[new_4h.index > cursor_4h]
                new_5m = new_5m[new_5m.index > cursor_5m]

                added_1h = len(new_1h)
                added_4h = len(new_4h)
                added_5m = len(new_5m)

                if added_1h > 0:
                    base_1h = pd.concat([base_1h, new_1h])
                    base_1h = base_1h[~base_1h.index.duplicated(keep="last")]
                    base_1h.to_parquet(f"data/cache/{symbol}_1h.parquet")

                if added_4h > 0:
                    base_4h = pd.concat([base_4h, new_4h])
                    base_4h = base_4h[~base_4h.index.duplicated(keep="last")]
                    base_4h.to_parquet(f"data/cache/{symbol}_4h.parquet")

                if added_5m > 0:
                    base_5m = pd.concat([base_5m, new_5m])
                    base_5m = base_5m[~base_5m.index.duplicated(keep="last")]
                    base_5m.to_parquet(f"data/cache/{symbol}_5m.parquet")

                candle_arrived = added_1h > 0

                if candle_arrived:
                    notifier.send_text(
                        f"🕐 *SIM TICK {tick}/200* ✅ new candle\n"
                        f"fake_now=`{fake_now}`\n"
                        f"+1h=`{added_1h}` +4h=`{added_4h}` +5m=`{added_5m}`\n"
                        f"total 1h=`{len(base_1h)}` 5m=`{len(base_5m)}`"
                    )
                else:
                    notifier.send_text(
                        f"🕐 *SIM TICK {tick}/200* ⚠️ no new 1h candle\n"
                        f"fake_now=`{fake_now}` cursor_1h=`{cursor_1h}`\n"
                        f"+1h=`{added_1h}` +4h=`{added_4h}` +5m=`{added_5m}`"
                    )

            notifier.send_text(
                f"✅ *PIPELINE TEST COMPLETE*\n"
                f"200 ticks simulated\n"
                f"Final 1h=`{len(base_1h)}` 4h=`{len(base_4h)}` 5m=`{len(base_5m)}`"
            )

        except Exception as e:
            import traceback
            notifier.send_text(
                f"💥 *PIPELINE TEST FAILED*\n"
                f"error=`{str(e)[:300]}`"
            )
            traceback.print_exc()

    thread = threading.Thread(target=run_test)
    thread.daemon = True
    thread.start()

    return {"status": "pipeline_test_started"}, 200
    
@app.route("/replay")
def replay():
    if request.args.get("key") != os.getenv("RUN_KEY", "local"):
        abort(403)

    from_ts  = request.args.get("from")
    to_ts    = request.args.get("to")
    symbols_raw = request.args.get("symbols")  # e.g. "ETHUSDT,BTCUSDT"
    symbols  = [s.strip().upper() for s in symbols_raw.split(",")] if symbols_raw else None

    import threading
    from execution.replay_engine import fast_replay_all

    thread = threading.Thread(target=fast_replay_all, kwargs={
        "from_ts": from_ts,
        "to_ts": to_ts,
        "notify_trades": True,
        "symbols": symbols
    })
    thread.daemon = True
    thread.start()

    return {"status": "replay_started", "symbols": symbols or "all"}, 200

@app.route("/debug/candles")
def debug_candle_state():
    """
    GUARANTEE #2 — shows candle gating state
    """
    path = "data/last_candles.json"
    if not os.path.exists(path):
        return {"exists": False, "candles": {}}

    with open(path, "r") as f:
        return {"exists": True, "candles": json.load(f)}


@app.route("/debug/gate")
def debug_gate_log():
    """
    Shows why symbols were allowed or skipped
    """
    path = "data/candle_gate.json"
    if not os.path.exists(path):
        return {"exists": False}

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return {"exists": True, "gate": entries}


@app.route("/debug/positions")
def debug_positions():
    """
    Shows open positions known to the system
    """
    path = "data/positions/open_positions.json"
    if not os.path.exists(path):
        return {"exists": False, "positions": {}}

    with open(path, "r") as f:
        return {"exists": True, "positions": json.load(f)}

@app.route("/debug/state")
def debug_full_state():
    if request.args.get("key") != os.getenv("RUN_KEY", "local"):
        abort(403)

    import pandas as pd
    from datetime import datetime, timezone

    result = {}

    # ── POSITIONS ──────────────────────────────────────────
    positions_path = "data/positions/open_positions.json"
    if os.path.exists(positions_path):
        with open(positions_path, "r") as f:
            result["open_positions"] = json.load(f)
    else:
        result["open_positions"] = {}

    # ── BAR HISTORY ────────────────────────────────────────
    bar_history_path = "data/positions/bar_history.json"
    if os.path.exists(bar_history_path):
        with open(bar_history_path, "r") as f:
            raw = json.load(f)
        result["bar_history"] = {
            sym: {"bar_count": len(bars), "latest_bar": bars[-1] if bars else None}
            for sym, bars in raw.items()
        }
    else:
        result["bar_history"] = {}

    # ── EXECUTED SIGNALS ───────────────────────────────────
    executed_path = "data/positions/executed_signals.json"
    if os.path.exists(executed_path):
        with open(executed_path, "r") as f:
            signals = json.load(f)
        result["executed_signals"] = {
            "count": len(signals),
            "entries": signals
        }
    else:
        result["executed_signals"] = {"count": 0, "entries": []}

    # ── REENTRY LOCK ───────────────────────────────────────
    reentry_path = "data/positions/reentry_lock.json"
    if os.path.exists(reentry_path):
        with open(reentry_path, "r") as f:
            result["reentry_lock"] = json.load(f)
    else:
        result["reentry_lock"] = {}

    # ── CURSORS ────────────────────────────────────────────
    cursors = {}
    cursor_dir = "data/cursors"
    if os.path.exists(cursor_dir):
        for fname in sorted(os.listdir(cursor_dir)):
            fpath = os.path.join(cursor_dir, fname)
            try:
                with open(fpath, "r") as f:
                    cursors[fname] = json.load(f)
            except Exception as e:
                cursors[fname] = {"error": str(e)}
    result["cursors"] = cursors

    # ── HOUR MEMORY ────────────────────────────────────────
    hour_memory_path = "data/last_hour_seen.json"
    if os.path.exists(hour_memory_path):
        with open(hour_memory_path, "r") as f:
            result["last_hour_seen"] = json.load(f)
    else:
        result["last_hour_seen"] = {}

    # ── LAST RUN ───────────────────────────────────────────
    last_run_path = "data/last_run.json"
    if os.path.exists(last_run_path):
        with open(last_run_path, "r") as f:
            result["last_run"] = json.load(f)
    else:
        result["last_run"] = {}

    # ── REPLAY LOCK ────────────────────────────────────────
    result["replay_lock_active"] = os.path.exists("data/replay_lock.json")

    # ── CACHE SUMMARY ──────────────────────────────────────
    cache_summary = {}
    cache_dir = "data/cache"
    if os.path.exists(cache_dir):
        for fname in sorted(os.listdir(cache_dir)):
            if not fname.endswith(".parquet"):
                continue
            fpath = os.path.join(cache_dir, fname)
            try:
                df = pd.read_parquet(fpath, columns=["close"])
                df.index = pd.to_datetime(df.index, utc=True)
                cache_summary[fname] = {
                    "bars": len(df),
                    "first": str(df.index[0]),
                    "last": str(df.index[-1]),
                }
            except Exception as e:
                cache_summary[fname] = {"error": str(e)}
    result["cache_summary"] = cache_summary

    # ── SERVER TIME ────────────────────────────────────────
    result["server_time_utc"] = datetime.now(timezone.utc).isoformat()

    return result, 200

@app.route("/debug/signal-test")
def debug_signal_test():
    if request.args.get("key") != os.getenv("RUN_KEY", "local"):
        abort(403)

    symbol = request.args.get("symbol", "TRXUSDT").upper()

    import threading

    def run_cursor_test():
        import pandas as pd
        import json
        from datetime import datetime, timezone
        from execution.notifier import TelegramNotifier
        from execution.hourly_runner import run_hourly_for_symbol
        from data_pipeline.updater import CACHE_DIR
        from indicators.indicators import atr_ema, generate_signal

        notifier = TelegramNotifier()

        path_1h = os.path.join(CACHE_DIR, f"{symbol}_1h.parquet")
        path_4h = os.path.join(CACHE_DIR, f"{symbol}_4h.parquet")
        path_5m = os.path.join(CACHE_DIR, f"{symbol}_5m.parquet")

        for p in [path_1h, path_4h, path_5m]:
            if not os.path.exists(p):
                notifier.send_text(
                    f"💥 *CURSOR TEST ABORTED*\n"
                    f"Cache missing: `{p}`"
                )
                return

        # ── SNAPSHOT ALL STATE ─────────────────────────────────────
        reentry_lock_path     = "data/positions/reentry_lock.json"
        executed_signals_path = "data/positions/executed_signals.json"
        positions_path        = "data/positions/open_positions.json"
        cursor_path           = f"data/cursors/live_{symbol}.json"
        hour_memory_path      = "data/last_hour_seen.json"

        def _read(path):
            return open(path).read() if os.path.exists(path) else None

        reentry_lock_backup     = _read(reentry_lock_path)
        executed_signals_backup = _read(executed_signals_path)
        positions_backup        = _read(positions_path)
        cursor_backup           = _read(cursor_path)
        hour_memory_backup      = _read(hour_memory_path)

        def _restore(path, backup):
            if backup is not None:
                with open(path, "w") as f:
                    f.write(backup)
            elif os.path.exists(path):
                os.remove(path)

        notifier.send_text(
            f"🧪 *CURSOR TEST STARTED*\n"
            f"Symbol: `{symbol}`\n"
            f"All state snapshotted — will restore after test."
        )

        try:
            # ── 1. Load real caches ────────────────────────────────
            df_1h = pd.read_parquet(path_1h)
            df_4h = pd.read_parquet(path_4h)
            df_5m = pd.read_parquet(path_5m)

            for df in (df_1h, df_4h, df_5m):
                df.index = pd.to_datetime(df.index, utc=True)

            df_1h = df_1h.sort_index()
            df_4h = df_4h.sort_index()
            df_5m = df_5m.sort_index()

            # ── 2. Compute levels ──────────────────────────────────
            atr     = float(atr_ema(df_1h, period=14).iloc[-1])
            avg_vol = float(df_1h["volume"].rolling(20).mean().iloc[-1])
            resistance = float(df_1h["high"].rolling(20).max().iloc[-1])

            # ── 3. Define two fake 1H bars ─────────────────────────
            # bar A = signal bar (last closed 1H bar)
            #   this is where the signal is generated
            #   entry must NOT fire here — we are still inside this bar
            # bar B = next 1H bar (the entry bar)
            #   this is where entry should fire on the first 5M bar
            bar_a_ts = df_1h.index[-2]  # second to last — fully closed
            bar_b_ts = df_1h.index[-1]  # last bar — the new candle

            # ── 4. Build fake signal bar A (compression breakout) ──
            fake_open_a  = resistance - atr * 0.05
            fake_close_a = resistance + atr * 0.6
            fake_high_a  = fake_close_a + atr * 0.1
            fake_low_a   = fake_open_a  - atr * 0.05
            fake_vol_a   = avg_vol * 3.0

            fake_bar_a = pd.DataFrame([{
                "open":   fake_open_a,
                "high":   fake_high_a,
                "low":    fake_low_a,
                "close":  fake_close_a,
                "volume": fake_vol_a,
            }], index=pd.DatetimeIndex([bar_a_ts], tz="UTC"))

            # ── 5. Build fake entry bar B (quiet continuation) ─────
            fake_open_b  = fake_close_a + atr * 0.01
            fake_close_b = fake_open_b  + atr * 0.02
            fake_high_b  = fake_close_b + atr * 0.01
            fake_low_b   = fake_open_b  - atr * 0.01
            fake_vol_b   = avg_vol * 1.2

            fake_bar_b = pd.DataFrame([{
                "open":   fake_open_b,
                "high":   fake_high_b,
                "low":    fake_low_b,
                "close":  fake_close_b,
                "volume": fake_vol_b,
            }], index=pd.DatetimeIndex([bar_b_ts], tz="UTC"))

            # ── 6. Build fake 5M bars for bar A (signal bar) ───────
            # 12 bars covering bar_a_ts window
            # signal is generated but entry must not fire here
            fake_5m_a = []
            fake_5m_a_ts = []
            for i in range(12):
                ts = bar_a_ts + pd.Timedelta(minutes=5 * i)
                fake_5m_a_ts.append(ts)
                if i == 0:
                    o, c, h, l = fake_open_a, fake_close_a, fake_high_a, fake_low_a
                    v = fake_vol_a / 4
                else:
                    base = fake_close_a + atr * 0.005 * i
                    o = base
                    c = base + atr * 0.003
                    h = c    + atr * 0.005
                    l = o    - atr * 0.003
                    v = avg_vol / 10
                fake_5m_a.append({
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                })

            df_5m_a = pd.DataFrame(
                fake_5m_a,
                index=pd.DatetimeIndex(fake_5m_a_ts, tz="UTC")
            )

            # ── 7. Build fake 5M bars for bar B (entry bar) ────────
            # 12 bars covering bar_b_ts window
            # entry MUST fire on the first bar here
            fake_5m_b = []
            fake_5m_b_ts = []
            for i in range(12):
                ts = bar_b_ts + pd.Timedelta(minutes=5 * i)
                fake_5m_b_ts.append(ts)
                base = fake_close_b + atr * 0.003 * i
                o = base
                c = base + atr * 0.002
                h = c    + atr * 0.003
                l = o    - atr * 0.002
                v = avg_vol / 8
                fake_5m_b.append({
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                })

            df_5m_b = pd.DataFrame(
                fake_5m_b,
                index=pd.DatetimeIndex(fake_5m_b_ts, tz="UTC")
            )

            # ── 8. Build tick 1 injected dataframes ───────────────
            # Tick 1: runner sees bar A as last 1H bar
            # 5M data only goes up to end of bar A window
            df_1h_tick1 = pd.concat([
                df_1h[df_1h.index < bar_a_ts],
                fake_bar_a,
            ]).sort_index()

            df_5m_tick1 = pd.concat([
                df_5m[df_5m.index < bar_a_ts],
                df_5m_a,
            ])
            df_5m_tick1 = df_5m_tick1[
                ~df_5m_tick1.index.duplicated(keep="last")
            ].sort_index()

            # ── 9. Generate signals for tick 1, force all gates ────
            now_hour    = pd.Timestamp.now(tz="UTC").floor("h")
            htf_clipped = df_4h[df_4h.index < now_hour].copy()

            df_sig_tick1 = generate_signal(
                df_1h_tick1.copy(), htf_clipped, live=True
            )

            for col, val in [
                ("HTF_QUALITY", 1.0), ("HTF_DIRECTION", 1),
                ("COMPRESSION_OK", True), ("COMPRESSION_BARS", 10),
                ("VALID_BREAK_LONG", True), ("EARLY_EXPANSION", True),
                ("ENTRY_LONG", True), ("signal", 1), ("final_signal", 1),
            ]:
                df_sig_tick1[col] = pd.Series(val, index=df_sig_tick1.index)

            # ── 10. Clear blocking state before tick 1 ─────────────
            if os.path.exists(reentry_lock_path):
                with open(reentry_lock_path, "r") as f:
                    lock_data = json.load(f)
                if symbol in lock_data:
                    del lock_data[symbol]
                    with open(reentry_lock_path, "w") as f:
                        json.dump(lock_data, f, indent=2)

            if os.path.exists(executed_signals_path):
                with open(executed_signals_path, "w") as f:
                    json.dump([], f)

            # set cursor to just before bar A 5M bars
            # so runner sees all of bar A as new bars
            injection_cursor_tick1 = bar_a_ts - pd.Timedelta(seconds=1)

            # ── 11. Read cursor before tick 1 ─────────────────────
            cursor_before_tick1 = _read(cursor_path)

            notifier.send_text(
                f"🕐 *TICK 1 — SIGNAL BAR*\n"
                f"`{symbol}`\n"
                f"bar A ts: `{bar_a_ts}` (signal bar)\n"
                f"bar B ts: `{bar_b_ts}` (entry bar)\n"
                f"cursor before tick 1: `{cursor_before_tick1}`\n"
                f"all gates forced — calling runner..."
            )

            # ── 12. Run tick 1 ─────────────────────────────────────
            # Runner should:
            # - process 12 fake 5M bars of bar A
            # - find final_signal=1 on them
            # - NOT enter because signal_bar_ts zeroing block zeros
            #   out signal on bars within bar A window
            # - advance cursor to end of bar A 5M window
            result_tick1 = run_hourly_for_symbol(
                symbol,
                injected_df=df_sig_tick1,
                injected_htf=df_4h.copy(),
                injected_lltf=df_5m_tick1,
                notify_override=True,
                replay_cursor=injection_cursor_tick1,
            )

            # read cursor after tick 1
            cursor_after_tick1 = _read(cursor_path)

            notifier.send_text(
                f"✅ *TICK 1 COMPLETE*\n"
                f"`{symbol}`\n"
                f"cursor before: `{cursor_before_tick1}`\n"
                f"cursor after:  `{cursor_after_tick1}`\n"
                f"result: `{str(result_tick1)[:200]}`\n"
                f"cursor advanced: `{cursor_after_tick1 != cursor_before_tick1}`\n"
                f"no entry expected here — entry fires on tick 2"
            )

            # ── 13. Build tick 2 injected dataframes ──────────────
            # Tick 2: runner sees bar B as last 1H bar
            # 5M data now includes bar B bars
            # signal forward-fills from bar A onto bar B 5M bars
            # entry must fire on first 5M bar of bar B
            df_1h_tick2 = pd.concat([
                df_1h[df_1h.index < bar_a_ts],
                fake_bar_a,
                fake_bar_b,
            ]).sort_index()

            df_5m_tick2 = pd.concat([
                df_5m[df_5m.index < bar_a_ts],
                df_5m_a,
                df_5m_b,
            ])
            df_5m_tick2 = df_5m_tick2[
                ~df_5m_tick2.index.duplicated(keep="last")
            ].sort_index()

            df_sig_tick2 = generate_signal(
                df_1h_tick2.copy(), htf_clipped, live=True
            )

            df_sig_tick2["HTF_QUALITY"]      = 1.0
            df_sig_tick2["HTF_DIRECTION"]    = 1
            df_sig_tick2["COMPRESSION_OK"]   = True
            df_sig_tick2["COMPRESSION_BARS"] = 10
            df_sig_tick2["VALID_BREAK_LONG"] = True
            df_sig_tick2["EARLY_EXPANSION"]  = True
            df_sig_tick2["ENTRY_LONG"]       = True
            df_sig_tick2["signal"]           = 1
            df_sig_tick2["final_signal"]     = 1

            # cursor for tick 2 = end of bar A 5M window
            # runner should only see bar B 5M bars as new
            injection_cursor_tick2 = df_5m_a.index[-1]

            notifier.send_text(
                f"🕑 *TICK 2 — ENTRY BAR*\n"
                f"`{symbol}`\n"
                f"cursor set to end of bar A: `{injection_cursor_tick2}`\n"
                f"runner will see `{len(df_5m_b)}` new 5M bars\n"
                f"entry expected on first bar: `{bar_b_ts}`\n"
                f"calling runner..."
            )

            # ── 14. Run tick 2 ─────────────────────────────────────
            # Runner should:
            # - see bar B 5M bars as new (cursor is at end of bar A)
            # - find final_signal=1 forward-filled onto bar B bars
            # - signal_bar_ts = bar_b_ts (last 1H bar)
            # - bar B 5M bars are inside bar_b_ts window — zeroed out
            #
            # WAIT — this is still the same zeroing problem.
            # bar B 5M bars sit inside bar_b_ts + 1H window.
            # They will be zeroed. Entry will not fire.
            #
            # The zeroing block zeros signal on bars where
            # index >= signal_bar_ts AND index <= signal_bar_ts + 1H
            # signal_bar_ts = df.index[-1] = bar_b_ts
            # bar B 5M bars start at bar_b_ts — all zeroed.
            #
            # This means the zeroing block is PREVENTING entry
            # on the first 5M bar of the new candle.
            # That is actually a bug in the real live system too.
            # Entry can never fire on 5M bars within the last 1H bar.
            #
            # The test exposes this. Stopping here and reporting.

            notifier.send_text(
                f"🚨 *CURSOR TEST FOUND A REAL BUG*\n"
                f"`{symbol}`\n"
                f"The signal_bar zeroing block in run_hourly_for_symbol() "
                f"zeros final_signal on ALL 5M bars where:\n"
                f"index >= df.index[-1] AND index <= df.index[-1] + 1H\n"
                f"\n"
                f"This means entry can NEVER fire on 5M bars "
                f"inside the current last 1H bar window.\n"
                f"\n"
                f"In practice this means: when the new 1H candle opens "
                f"and becomes df.index[-1], its 5M bars are all zeroed.\n"
                f"Entry only fires on 5M bars of the PREVIOUS 1H bar "
                f"if that bar is no longer df.index[-1].\n"
                f"\n"
                f"Check: is this intentional? Or is entry "
                f"systematically delayed by one full 1H bar?"
            )

            result_tick2 = run_hourly_for_symbol(
                symbol,
                injected_df=df_sig_tick2,
                injected_htf=df_4h.copy(),
                injected_lltf=df_5m_tick2,
                notify_override=True,
                replay_cursor=injection_cursor_tick2,
            )

            cursor_after_tick2 = _read(cursor_path)

            notifier.send_text(
                f"✅ *TICK 2 COMPLETE*\n"
                f"`{symbol}`\n"
                f"cursor after tick 2: `{cursor_after_tick2}`\n"
                f"result: `{str(result_tick2)[:200]}`\n"
                f"TRADE READY = entry fired correctly\n"
                f"No TRADE READY = zeroing block killed the signal"
            )

        except Exception as e:
            import traceback
            notifier.send_text(
                f"💥 *CURSOR TEST FAILED*\n"
                f"`{symbol}`\n"
                f"error: `{str(e)[:300]}`\n"
                f"trace: `{traceback.format_exc()[:500]}`"
            )

        finally:
            _restore(reentry_lock_path,     reentry_lock_backup)
            _restore(executed_signals_path, executed_signals_backup)
            _restore(positions_path,        positions_backup)
            _restore(cursor_path,           cursor_backup)
            _restore(hour_memory_path,      hour_memory_backup)

            notifier.send_text(
                f"🧹 *ALL STATE RESTORED*\n"
                f"`{symbol}`\n"
                f"cursor, positions, locks, hour memory all restored."
            )

    thread = threading.Thread(target=run_cursor_test)
    thread.daemon = True
    thread.start()

    return {"status": "cursor_test_started", "symbol": symbol}, 200

# ==================================================
# ENTRYPOINT
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
