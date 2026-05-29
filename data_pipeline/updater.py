import os
import pandas as pd
import time
from datetime import datetime, timezone, timedelta

from data_pipeline.fetcher import fetch_ohlcv
from data_pipeline.validators import validate_ohlcv
from execution.notifier import TelegramNotifier


CACHE_DIR = "data/cache"

HOURS_LOOKBACK = 1000

LLTF_INTERVAL = "5m"
LTF_INTERVAL = "1h"
HTF_INTERVAL = "4h"


def _now_utc_hour():
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, tf: str):
    return os.path.join(CACHE_DIR, f"{symbol}_{tf}.parquet")

def _fetch_all(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Paginating fetch — works backwards from end until start is covered.
    Handles any window size, bypassing Binance's 1000-bar per request limit.
    """
    all_chunks = []
    current_end = end

    while True:
        df = fetch_ohlcv(
            symbol   = symbol,
            interval = interval,
            start    = start,
            end      = current_end,
            limit    = 1000,
            verbose  = False,
        )

        if df.empty:
            break

        all_chunks.insert(0, df)

        # If we've reached or passed the start, we're done
        if df.index[0] <= start:
            break

        # Step back: next fetch ends just before the earliest bar we got
        current_end = df.index[0] - pd.Timedelta(milliseconds=1)
        time.sleep(0.25)   # stay within Binance rate limits

    if not all_chunks:
        return pd.DataFrame()

    result = pd.concat(all_chunks)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()
    return result

def update_symbol(symbol: str):

    print(f"\n========== UPDATE {symbol} ==========")

    _ensure_cache_dir()

    path_ltf  = _cache_path(symbol, LTF_INTERVAL)
    path_htf  = _cache_path(symbol, HTF_INTERVAL)
    path_lltf = _cache_path(symbol, LLTF_INTERVAL)

    now = datetime.now(timezone.utc)  # full timestamp for boundary check
    now_hour = now.replace(minute=0, second=0, microsecond=0)
    start_required = now_hour - timedelta(hours=HOURS_LOOKBACK)

    # Compute 4H boundary once — used in both fast-exit and slow paths
    hours_into_cycle = now_hour.hour % 4
    current_4h_open = now_hour - timedelta(hours=hours_into_cycle)

    # --------------------------------------------------
    # FAST EARLY-EXIT — nothing new to fetch
    # --------------------------------------------------
    if os.path.exists(path_lltf) and os.path.getsize(path_lltf) > 0:
        try:
            df_check = pd.read_parquet(path_lltf, columns=["close"])
            df_check.index = pd.to_datetime(df_check.index, utc=True)
            last_5m_ts = df_check.index[-1]

            minutes_floored = (now.minute // 5) * 5
            current_5m_boundary = now.replace(minute=minutes_floored, second=0, microsecond=0)
            candle_age_seconds = (now - current_5m_boundary).total_seconds()

            # wait at least 10 seconds after candle close before processing
            # prevents acting on unclosed or not-yet-propagated candles
            if last_5m_ts >= current_5m_boundary:
                if candle_age_seconds >= 10:
                    print(f"[SKIP] {symbol} — cache is current (last={last_5m_ts}, boundary={current_5m_boundary})")
                else:
                    print(f"[CANDLE FRESH] {symbol} — {candle_age_seconds:.0f}s since close, waiting for propagation")

                # check if 1H cache has the latest closed candle
                ltf_check = pd.read_parquet(path_ltf)
                ltf_check.index = pd.to_datetime(ltf_check.index, utc=True)
                if ltf_check.index[-1] < now_hour - timedelta(hours=1):
                    print(f"[SKIP BYPASSED] {symbol} — 1H cache behind ({ltf_check.index[-1]} < {now_hour - timedelta(hours=1)}), fetching")
                    raise Exception("1H cache stale — force full fetch")

                # also check 5m cache isn't stale by more than 2 bars
                expected_5m = current_5m_boundary
                actual_5m = last_5m_ts
                if (expected_5m - actual_5m).total_seconds() > 600:  # more than 2 bars behind
                    print(f"[SKIP BYPASSED] {symbol} — 5m cache stale by {(expected_5m - actual_5m).total_seconds()/60:.0f}m, fetching")
                    raise Exception("5m cache stale — force full fetch")

                df_lltf = pd.read_parquet(path_lltf)
                df_lltf.index = pd.to_datetime(df_lltf.index, utc=True)
                df = ltf_check
                df_htf  = pd.read_parquet(path_htf)
                df_htf.index  = pd.to_datetime(df_htf.index,  utc=True)
                df = df[df.index <= now_hour - timedelta(hours=1)]
                hours_into_cycle = now_hour.hour % 4
                _last_closed_4h = now_hour - timedelta(hours=hours_into_cycle) - timedelta(hours=4)
                _current_4h_open = _last_closed_4h + timedelta(hours=4)
                df_htf = df_htf[df_htf.index < _current_4h_open]

                # Load HTF scores cache for fast-exit path
                _htf_scores = None
                _path_htf_scores = _cache_path(symbol, "htf_scores")
                if os.path.exists(_path_htf_scores):
                    try:
                        _htf_scores = pd.read_parquet(_path_htf_scores)
                        _htf_scores.index = pd.to_datetime(_htf_scores.index, utc=True)
                    except Exception:
                        _htf_scores = None

                return df, df_htf, df_lltf, _htf_scores
        except Exception as e:
            import pyarrow.lib as _pal
            # ArrowInvalid inherits from ValueError — must check isinstance
            # before any isinstance(e, ValueError) branch or it gets swallowed.
            _is_file_error = isinstance(
                e, (OSError, PermissionError, MemoryError,
                    _pal.ArrowInvalid, _pal.ArrowIOError)
            )
            if _is_file_error:
                print(f"[CACHE CORRUPT] {symbol} — {type(e).__name__}: {e}")
                for _p in [path_lltf, path_ltf]:
                    try:
                        if os.path.exists(_p):
                            os.remove(_p)
                            print(f"[CACHE CORRUPT] deleted {_p}")
                    except Exception:
                        pass
                try:
                    TelegramNotifier().send_text(
                        f"⚠️ *CACHE FILE ERROR*\n"
                        f"`{symbol}` `{type(e).__name__}`\n"
                        f"`{str(e)[:200]}`"
                    )
                except Exception:
                    pass
            elif isinstance(e, (ValueError, KeyError, IndexError)):
                print(f"[SKIP CHECK BYPASSED] {symbol} — {e}, fetching")
            else:
                print(f"[SKIP FAILED] {symbol} — {type(e).__name__}: {e}")
            # always fall through to full fetch

    now_full = now   # preserve full-precision timestamp for 5m fetch
    now = now_hour   # 1H and 4H fetches use top-of-hour only

    df = None
    last_ts = None

    # --------------------------------------------------
    # LOAD CACHE
    # --------------------------------------------------

    if os.path.exists(path_ltf):
        if os.path.getsize(path_ltf) == 0:
            print(f"[WARN] LTF cache is 0 bytes, discarding: {path_ltf}")
            os.remove(path_ltf)
        else:
            print("[CACHE] Loading LTF cache")
            df = pd.read_parquet(path_ltf)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            if not df.empty:
                last_ts = df.index[-1]

    # --------------------------------------------------
    # DETERMINE FETCH WINDOW
    # --------------------------------------------------

    fetch_start = start_required if df is None else last_ts + timedelta(hours=1)
    fetch_end = now_hour - timedelta(hours=1)  # only fetch closed 1H bars

    print("[FETCH WINDOW]")
    print("start:", fetch_start)
    print("end:", fetch_end)

    # --------------------------------------------------
    # FETCH NEW DATA
    # --------------------------------------------------

    if fetch_start <= fetch_end:
        new_data = _fetch_all(symbol, LTF_INTERVAL, fetch_start, fetch_end)

        if not new_data.empty:

            print("[MERGE] merging new candles")

            df = pd.concat([df, new_data]) if df is not None else new_data
            df = df[~df.index.duplicated(keep="last")]
    
    if df is None or df.empty:
        raise RuntimeError(f"[{symbol}] No LTF data available after fetch")

    # --------------------------------------------------
    # FINAL CLEAN
    # --------------------------------------------------

    df = df.sort_index()
    df = df[df.index >= start_required]
    df = df[df.index <= now_hour - timedelta(hours=1)]  # only closed bars before gap check and save
    df = df.iloc[-HOURS_LOOKBACK:]

    print("[DATA] final LTF candles:", len(df))

    # --------------------------------------------------
    # GAP CHECK
    # --------------------------------------------------

    # Floor to the interval frequency before comparison.
    # Binance occasionally returns candles with sub-millisecond timestamp
    # offsets (13:00:00.001 instead of 13:00:00.000). Without flooring,
    # symmetric_difference sees the offset bar as both missing and extra,
    # raises RuntimeError, and the symbol is dead until cache rebuilds.
    df.index = df.index.floor(LTF_INTERVAL)
    df = df[~df.index.duplicated(keep="last")]

    expected = pd.date_range(
        start=df.index[0],
        periods=len(df),
        freq=LTF_INTERVAL,
        tz="UTC"
    )

    if not df.index.equals(expected):

        diff = df.index.symmetric_difference(expected)

        raise RuntimeError(
            f"[{symbol}] LTF GAP DETECTED {diff[:5]}"
        )

    # --------------------------------------------------
    # VALIDATE LTF
    # --------------------------------------------------

    validate_ohlcv(df, symbol, freq=LTF_INTERVAL)

    # --------------------------------------------------
    # BUILD HTF (incremental, cache-aware)
    # --------------------------------------------------

    df_htf = None
    last_htf_ts = None

    # Load HTF cache if exists
    if os.path.exists(path_htf):
        if os.path.getsize(path_htf) == 0:
            print(f"[WARN] HTF cache is 0 bytes, discarding: {path_htf}")
            os.remove(path_htf)
        else:
            print("[CACHE] Loading HTF cache")
            df_htf = pd.read_parquet(path_htf)
            df_htf.index = pd.to_datetime(df_htf.index, utc=True)
            df_htf = df_htf.sort_index()
            if not df_htf.empty:
                last_htf_ts = df_htf.index[-1]

    # Determine fetch window
    htf_fetch_start = start_required if df_htf is None else last_htf_ts + timedelta(hours=4)
    htf_fetch_end = current_4h_open  # fetch up to but not including the open bar

    print("[FETCH HTF WINDOW]")
    print("start:", htf_fetch_start)
    print("end:", htf_fetch_end)

    # Fetch only missing HTF candles
    if htf_fetch_start <= htf_fetch_end:
        new_htf = _fetch_all(symbol, HTF_INTERVAL, htf_fetch_start, htf_fetch_end)

        if not new_htf.empty:
            print("[MERGE HTF] merging new candles")
            df_htf = pd.concat([df_htf, new_htf]) if df_htf is not None else new_htf
            df_htf = df_htf[~df_htf.index.duplicated(keep="last")]
    
    if df_htf is None or df_htf.empty:
        raise RuntimeError(f"[{symbol}] No HTF data available after fetch")

    # Only keep closed 4H bars.
    # A 4H bar that opened at T is closed when now >= T + 4h.
    # last_closed_4h = the open timestamp of the most recent fully closed 4H bar.
    # Example: now_hour=21:00 → 21%4=1 → last_closed_4h = 21:00 - 1h - 4h = 16:00 ✓
    # Example: now_hour=20:00 → 20%4=0 → last_closed_4h = 20:00 - 0h - 4h = 16:00 ✓
    # The 16:00 bar closes exactly at 20:00 — we exclude it at the boundary to be safe.

    df_htf = df_htf.sort_index()
    df_htf = df_htf[df_htf.index >= start_required]
    df_htf = df_htf[df_htf.index < current_4h_open]  # exclude open bar — prevents HTF_QUALITY drift
    df_htf = df_htf.iloc[-HOURS_LOOKBACK:]
    validate_ohlcv(df_htf, symbol, freq=HTF_INTERVAL)

    print(f"[DEBUG] live htf_df last={df_htf.index[-1]} len={len(df_htf)} current_4h_open={current_4h_open}")
    # TEMP DIAGNOSTIC
    for _ts, _row in df_htf.tail(3).iterrows():
        print(f"[DEBUG HTF BAR] {_ts} | open={_row['open']:.4f} close={_row['close']:.4f} volume={_row['volume']:.2f}")

    print("[HTF] candles:", len(df_htf))

    # --------------------------------------------------
    # HTF SCORES CACHE (compute once per 4H close)
    # --------------------------------------------------
    from indicators.indicators import compute_htf_scores

    path_htf_scores = _cache_path(symbol, "htf_scores")
    htf_scores = None
    last_scores_ts = None

    if os.path.exists(path_htf_scores):
        try:
            htf_scores = pd.read_parquet(path_htf_scores)
            htf_scores.index = pd.to_datetime(htf_scores.index, utc=True)
            last_scores_ts = htf_scores.index[-1]
        except Exception as e:
            print(f"[HTF SCORES] cache load failed: {e}, recomputing")
            htf_scores = None
            last_scores_ts = None

    htf_last_ts = df_htf.index[-1]

    if last_scores_ts is None or last_scores_ts < htf_last_ts:
        print(f"[HTF SCORES] recomputing — scores_last={last_scores_ts} htf_last={htf_last_ts}")
        htf_scores = compute_htf_scores(df_htf)

        tmp_scores = path_htf_scores + ".tmp"
        htf_scores.to_parquet(tmp_scores)
        os.replace(tmp_scores, path_htf_scores)
        print(f"[HTF SCORES] saved — {len(htf_scores)} bars, last={htf_scores.index[-1]}")
    else:
        print(f"[HTF SCORES] cache current — last={last_scores_ts}")

    # --------------------------------------------------
    # SAVE ATOMIC
    # --------------------------------------------------
    os.makedirs(CACHE_DIR, exist_ok=True)

    tmp_ltf = path_ltf + ".tmp"
    tmp_htf = path_htf + ".tmp"

    df.to_parquet(tmp_ltf)
    df_htf.to_parquet(tmp_htf)

    os.makedirs(os.path.dirname(path_ltf), exist_ok=True)
    os.replace(tmp_ltf, path_ltf)
    os.makedirs(os.path.dirname(path_htf), exist_ok=True)
    os.replace(tmp_htf, path_htf)

    print("[SAVE] LTF + HTF cache updated")

    # --------------------------------------------------
    # BUILD LLTF (5M) — same pattern as HTF
    # --------------------------------------------------
    path_lltf    = _cache_path(symbol, LLTF_INTERVAL)
    df_lltf      = None
    last_lltf_ts = None

    if os.path.exists(path_lltf):
        if os.path.getsize(path_lltf) == 0:
            print(f"[WARN] LLTF cache is 0 bytes, discarding: {path_lltf}")
            os.remove(path_lltf)
        else:
            print("[CACHE] Loading LLTF cache")
            df_lltf = pd.read_parquet(path_lltf)
            df_lltf.index = pd.to_datetime(df_lltf.index, utc=True)
            df_lltf = df_lltf.sort_index()
            if not df_lltf.empty:
                last_lltf_ts = df_lltf.index[-1]

    lltf_fetch_start = start_required if df_lltf is None else last_lltf_ts + timedelta(minutes=5)
    lltf_fetch_end   = now_full  # 5m candles: use full-precision now so every cron fire fetches new bars

    print("[FETCH LLTF WINDOW]")
    print("start:", lltf_fetch_start)
    print("end:  ", lltf_fetch_end)

    if lltf_fetch_start <= lltf_fetch_end:
        new_lltf = _fetch_all(symbol, LLTF_INTERVAL, lltf_fetch_start, lltf_fetch_end)
        
        if not new_lltf.empty:
            print("[MERGE LLTF] merging new candles")
            df_lltf = pd.concat([df_lltf, new_lltf]) if df_lltf is not None else new_lltf
            df_lltf = df_lltf[~df_lltf.index.duplicated(keep="last")]

    if df_lltf is None or df_lltf.empty:
        raise RuntimeError(f"[{symbol}] No LLTF data available after fetch")

    df_lltf = df_lltf.sort_index()
    df_lltf = df_lltf[df_lltf.index >= start_required]
    df_lltf = df_lltf.iloc[-(HOURS_LOOKBACK * 12):]
    try:
        validate_ohlcv(df_lltf, symbol, freq=LLTF_INTERVAL)
    except RuntimeError as e:
        print(f"[WARN] LLTF validation failed for {symbol} (non-fatal): {e}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    tmp_lltf = path_lltf + ".tmp"
    df_lltf.to_parquet(tmp_lltf)
    os.makedirs(os.path.dirname(path_lltf), exist_ok=True)
    os.replace(tmp_lltf, path_lltf)

    print("[SAVE] LLTF cache updated | candles:", len(df_lltf))

    return df, df_htf, df_lltf, htf_scores