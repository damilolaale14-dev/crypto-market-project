import requests
import pandas as pd
from datetime import datetime, timezone
import time

BASE_URL = "https://api.binance.com/api/v3/klines"


def _to_ms(dt):
    """
    Convert a string, pandas Timestamp, or datetime to milliseconds since epoch UTC.
    """
    # Convert string to pandas Timestamp
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)

    # Convert pandas Timestamp to datetime
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    # Ensure datetime has UTC tzinfo
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        raise TypeError(f"_to_ms() expected str, pd.Timestamp, or datetime, got {type(dt)}")

    return int(dt.timestamp() * 1000)

def fetch_ohlcv(
    symbol: str,
    start: datetime = None,
    end: datetime = None,
    interval: str = "1h",
    limit: int = 1000,
    retries: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:

    symbol = symbol.replace("-", "").upper()

    def safe_request(params):
        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            # 🔥 HARD GUARD: Binance sometimes returns dict error payload
            if isinstance(data, dict):
                raise RuntimeError(f"Binance error response: {data}")

            # 🔥 HARD GUARD: empty or malformed
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected response type: {type(data)}")

            return data

        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")

    start_ms = _to_ms(start) if start else None
    end_ms = _to_ms(end) if end else None

    for attempt in range(retries):

        if attempt > 0:
            wait = 2 ** attempt  # 2s, 4s
            print(f"[FETCH RETRY] {symbol} attempt {attempt+1}/{retries} — waiting {wait}s")
            time.sleep(wait)

        try:
            if verbose:
                print(f"[FETCH] {symbol} | {interval}")
                print(f"[FETCH] start={start} end={end}")

            all_data = []
            end_time = end_ms

            while True:

                page_params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": 1000
                }

                if end_time:
                    page_params["endTime"] = end_time

                data = safe_request(page_params)

                if len(data) == 0:
                    break

                all_data = data + all_data

                oldest_open_time = data[0][0]
                end_time = oldest_open_time - 1

                # stop if we hit start boundary
                if start_ms and oldest_open_time <= start_ms:
                    break

                # safety stop for pagination correctness
                if len(data) < 1000:
                    break

                time.sleep(0.25)

            if not all_data:
                return pd.DataFrame()

            df = pd.DataFrame(all_data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_vol", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            df = df[["open", "high", "low", "close", "volume"]].astype(float)

            if verbose:
                print(f"[FETCH] returned {len(df)} candles")

            # final trim (safe even if pagination overshoots)
            if start_ms:
                df = df[df.index >= pd.to_datetime(start_ms, unit="ms", utc=True)]
            if end_ms:
                df = df[df.index <= pd.to_datetime(end_ms, unit="ms", utc=True)]

            return df

        except Exception as e:
            print(f"[FETCH ERROR] attempt {attempt+1}: {e}")
            # if rate limited, back off harder
            if "429" in str(e) or "418" in str(e):
                time.sleep(10 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch data for {symbol}")