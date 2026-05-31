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

@app.route("/test-proxy")
def test_proxy():
    import requests, os
    proxy_url = os.getenv("PROXY_URL")
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    try:
        r = requests.get("https://ipv4.webshare.io/", proxies=proxies, timeout=10)
        return {"proxy_url_set": bool(proxy_url), "your_ip": r.text.strip()}, 200
    except Exception as e:
        return {"error": str(e), "proxy_url_set": bool(proxy_url)}, 500
    
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

@app.route("/debug/signal-audit")
def debug_signal_audit():
    if request.args.get("key") != os.getenv("RUN_KEY", "local"):
        abort(403)

    symbol = request.args.get("symbol", "TRXUSDT").upper()

    import threading

    def run_audit():
        import pandas as pd
        from datetime import datetime, timezone
        from execution.notifier import TelegramNotifier
        from data_pipeline.updater import update_symbol, CACHE_DIR
        from indicators.indicators import generate_signal, atr_ema, map_ltf_to_htf
        from execution.hourly_runner import map_ltf_to_htf

        notifier = TelegramNotifier()

        try:
            notifier.send_text(
                f"🔍 *SIGNAL AUDIT STARTED*\n"
                f"Symbol: `{symbol}`\n"
                f"Using real live data — no injection"
            )

            # ── 1. Fetch exactly as the live runner does ───────────
            df, htf_df, lltf_df = update_symbol(symbol)

            for frame in (df, htf_df, lltf_df):
                frame.index = pd.to_datetime(frame.index, utc=True)

            # ── 2. Clip lltf exactly as runner does ────────────────
            now_utc = datetime.now(timezone.utc)
            minutes_floored = (now_utc.minute // 5) * 5
            current_5m_boundary = pd.Timestamp(
                now_utc.replace(minute=minutes_floored, second=0, microsecond=0)
            ).tz_convert("UTC")

            lltf_df = lltf_df[lltf_df.index < current_5m_boundary].copy()

            notifier.send_text(
                f"📦 *DATA LOADED*\n"
                f"`{symbol}`\n"
                f"1H bars: `{len(df)}` | last: `{df.index[-1]}`\n"
                f"5M bars: `{len(lltf_df)}` | last: `{lltf_df.index[-1]}`\n"
                f"current_5m_boundary: `{current_5m_boundary}`"
            )

            # ── 3. Generate signal exactly as runner does ──────────
            df_sig = generate_signal(df.copy(), htf_df.copy(), live=True)

            signal_bar_ts  = df_sig.index[-1]
            signal_bar_end = signal_bar_ts + pd.Timedelta(hours=1)
            last_signal    = int(df_sig['final_signal'].iloc[-1])
            last_20        = df_sig['final_signal'].iloc[-20:].tolist()

            notifier.send_text(
                f"📊 *SIGNAL STATE*\n"
                f"`{symbol}`\n"
                f"signal_bar_ts (df.index[-1]): `{signal_bar_ts}`\n"
                f"signal_bar_end (zeroing until): `{signal_bar_end}`\n"
                f"final_signal at signal bar: `{last_signal}`\n"
                f"last 20 final_signals: `{last_20}`"
            )

            # ── 4. Map 5M to 1H exactly as runner does ─────────────
            lltf_df = lltf_df[lltf_df.index >= df_sig.index[0]].copy()
            lltf_df = map_ltf_to_htf(lltf_df, df_sig)
            lltf_df['final_signal'] = df_sig['final_signal'].reindex(
                lltf_df.index, method='ffill'
            )
            lltf_df['ATR'] = df_sig['ATR'].reindex(lltf_df.index, method='ffill')

            # ── 5. Apply zeroing block exactly as runner does ───────
            within_signal_bar = (
                (lltf_df.index >= signal_bar_ts) &
                (lltf_df.index <= signal_bar_end)
            )
            zeroed_count = within_signal_bar.sum()
            lltf_df.loc[within_signal_bar, 'final_signal'] = 0

            notifier.send_text(
                f"🚫 *ZEROING BLOCK*\n"
                f"`{symbol}`\n"
                f"zeroing range: `{signal_bar_ts}` → `{signal_bar_end}`\n"
                f"5M bars zeroed: `{zeroed_count}`\n"
                f"5M bars surviving with signal: `{(lltf_df['final_signal'] != 0).sum()}`"
            )

            # ── 6. Find all 5M bars carrying a live signal ─────────
            signal_bars = lltf_df[lltf_df['final_signal'] != 0]

            if signal_bars.empty:
                notifier.send_text(
                    f"✅ *NO LIVE SIGNALS IN 5M STREAM*\n"
                    f"`{symbol}`\n"
                    f"Nothing would fire this tick — system is correctly gated\n"
                    f"If a signal exists on the 1H, it was zeroed correctly"
                )
            else:
                # For each signal bar, show which 1H bar it maps to
                # and whether it's inside or outside the signal bar window
                lines = []
                for ts, row in signal_bars.iterrows():
                    mapped_1h = df_sig.index[int(row['ltf_index'])]
                    inside_signal_window = (ts >= signal_bar_ts) and (ts <= signal_bar_end)
                    inside_entry_window  = (ts > signal_bar_end)
                    lines.append(
                        f"  5M: `{ts}` → 1H: `{mapped_1h}` | "
                        f"sig={int(row['final_signal'])} | "
                        f"{'⚠️ INSIDE SIGNAL BAR WINDOW' if inside_signal_window else '✅ after signal bar'}"
                    )

                first_valid = signal_bars.index[0]
                mapped_1h_first = df_sig.index[int(signal_bars.iloc[0]['ltf_index'])]
                is_same_bar = mapped_1h_first == signal_bar_ts
                is_next_bar = mapped_1h_first == signal_bar_ts + pd.Timedelta(hours=1)

                notifier.send_text(
                    f"⚡ *LIVE SIGNAL BARS FOUND*\n"
                    f"`{symbol}`\n"
                    f"total 5M bars with signal: `{len(signal_bars)}`\n"
                    f"first entry would fire at: `{first_valid}`\n"
                    f"that maps to 1H bar: `{mapped_1h_first}`\n"
                    f"signal bar was: `{signal_bar_ts}`\n"
                    f"verdict: `{'⚠️ SAME BAR — early entry bug' if is_same_bar else '✅ next bar — correct' if is_next_bar else '❓ neither — check manually'}`\n\n"
                    f"all signal 5M bars:\n" + "\n".join(lines[:20])
                )

            # ── 7. Load cursor and compare ─────────────────────────
            cursor_path = f"data/cursors/live_{symbol}.json"
            cursor_ts = None
            if os.path.exists(cursor_path):
                with open(cursor_path) as f:
                    raw = json.load(f)
                cursor_ts = pd.Timestamp(raw).tz_convert("UTC")

            lltf_frozen = lltf_df.copy().dropna(subset=['ltf_index'])
            new_bars = (
                lltf_frozen if cursor_ts is None
                else lltf_frozen[lltf_frozen.index > cursor_ts]
            )

            new_signal_bars = new_bars[new_bars['final_signal'] != 0]

            notifier.send_text(
                f"📍 *CURSOR vs NEW BARS*\n"
                f"`{symbol}`\n"
                f"cursor_ts: `{cursor_ts}`\n"
                f"lltf last: `{lltf_frozen.index[-1]}`\n"
                f"new bars since cursor: `{len(new_bars)}`\n"
                f"new bars carrying signal: `{len(new_signal_bars)}`\n"
                f"{'⚠️ signal bars exist in new window — entry would attempt' if not new_signal_bars.empty else '✅ no signal in new bars — no entry this tick'}\n"
                + (f"first signal bar in new window: `{new_signal_bars.index[0]}`" if not new_signal_bars.empty else "")
            )

        except Exception as e:
            import traceback
            notifier.send_text(
                f"💥 *SIGNAL AUDIT FAILED*\n"
                f"`{symbol}`\n"
                f"error: `{str(e)[:300]}`\n"
                f"trace: `{traceback.format_exc()[:500]}`"
            )

    thread = threading.Thread(target=run_audit)
    thread.daemon = True
    thread.start()

    return {"status": "signal_audit_started", "symbol": symbol}, 200


@app.route("/debug/cursor-health")
def debug_cursor_health():
    """
    Checks cursor state across all symbols and flags gaps.
    A 1-bar gap (5 minutes) is normal — cron hasn't fired yet.
    A 2+ bar gap means the cron fired but the cursor didn't advance,
    or update_symbol() returned early via fast-exit before processing.
    """
    import pandas as pd
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    minutes_floored = (now.minute // 5) * 5
    current_5m_boundary = pd.Timestamp(
        now.replace(minute=minutes_floored, second=0, microsecond=0)
    ).tz_convert("UTC")

    cursor_dir = "data/cursors"
    results = {}

    if not os.path.exists(cursor_dir):
        return {"error": "no cursor directory"}, 500

    for fname in sorted(os.listdir(cursor_dir)):
        if not fname.startswith("live_"):
            continue
        symbol = fname.replace("live_", "").replace(".json", "")
        fpath  = os.path.join(cursor_dir, fname)
        try:
            with open(fpath) as f:
                raw = json.load(f)
            cursor_ts = pd.Timestamp(raw).tz_convert("UTC")
            gap_seconds = (current_5m_boundary - cursor_ts).total_seconds()
            gap_bars    = int(gap_seconds // 300)  # 300s = 5 minutes
            results[symbol] = {
                "cursor_ts":            str(cursor_ts),
                "current_5m_boundary":  str(current_5m_boundary),
                "gap_bars":             gap_bars,
                "gap_minutes":          int(gap_seconds // 60),
                "status": (
                    "ok"       if gap_bars <= 1 else
                    "warning"  if gap_bars <= 3 else
                    "critical"
                ),
                "note": (
                    "normal — cron hasnt fired yet"         if gap_bars <= 1 else
                    "cron fired but cursor didnt advance"   if gap_bars <= 3 else
                    "cursor significantly behind — check fast-exit or cron health"
                ),
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}

    return {"server_time_utc": now.isoformat(), "cursors": results}, 200

# ==================================================
# ENTRYPOINT
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)