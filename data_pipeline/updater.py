import os
import pandas as pd
import time
from datetime import datetime, timezone, timedelta

from data_pipeline.fetcher import fetch_ohlcv
from data_pipeline.validators import validate_ohlcv#


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

                # return cached data in both cases — fresh candle path was
                # silently falling through to full fetch and rewriting parquets
                df      = pd.read_parquet(path_ltf)
                df_htf  = pd.read_parquet(path_htf)
                df_lltf = pd.read_parquet(path_lltf)
                df.index      = pd.to_datetime(df.index,      utc=True)
                df_htf.index  = pd.to_datetime(df_htf.index,  utc=True)
                df_lltf.index = pd.to_datetime(df_lltf.index, utc=True)
                # strip any forming candle so generate_signal never sees partial data
                df     = df[df.index < now_hour]
                df_htf = df_htf[df_htf.index < now_hour]
                return df, df_htf, df_lltf
        except Exception as e:
            print(f"[SKIP CHECK FAILED] {symbol} — {e}, proceeding with full fetch")

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
    fetch_end = now_hour  # 1H candles: only fetch closed candles

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
    df = df.iloc[-HOURS_LOOKBACK:]

    print("[DATA] final LTF candles:", len(df))

    # --------------------------------------------------
    # GAP CHECK
    # --------------------------------------------------

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
    htf_fetch_end = now_hour  # 4H candles: only fetch closed candles

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

    df_htf = df_htf.sort_index()
    df_htf = df_htf[df_htf.index >= start_required]
    df_htf = df_htf.iloc[-HOURS_LOOKBACK:]
    validate_ohlcv(df_htf, symbol, freq=HTF_INTERVAL)

    # Final clean + validation
    if df_htf is not None:
        df_htf = df_htf.sort_index()
        df_htf = df_htf[df_htf.index >= start_required]
        df_htf = df_htf.iloc[-HOURS_LOOKBACK:]  # keep consistent history
        validate_ohlcv(df_htf, symbol, freq=HTF_INTERVAL)

    print("[HTF] candles:", len(df_htf))

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

    # strip forming candle before returning so generate_signal never sees partial 1H/4H data
    df     = df[df.index < now_hour]
    df_htf = df_htf[df_htf.index < now_hour]

    return df, df_htf, df_lltf