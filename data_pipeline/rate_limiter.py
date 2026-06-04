# data_pipeline/rate_limiter.py
import time
import json
import os

STATE_FILE = "data/rate_limiter_state.json"

class BinanceRateLimiter:
    def __init__(self):
        self.banned_until = 0
        self.rate_limited_until = 0
        self.current_weight = 0
        self._weight_window_start = time.time()
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.banned_until = state.get("banned_until", 0)
                self.rate_limited_until = state.get("rate_limited_until", 0)
                self.current_weight = state.get("current_weight", 0)
                self._weight_window_start = state.get("weight_window_start", time.time())
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE + ".tmp", "w") as f:
            json.dump({
                "banned_until": self.banned_until,
                "rate_limited_until": self.rate_limited_until,
                "current_weight": self.current_weight,
                "weight_window_start": self._weight_window_start,
            }, f)
        os.replace(STATE_FILE + ".tmp", STATE_FILE)

    def is_banned(self, buffer_secs=900) -> bool:
        self._load()
        return time.time() < self.banned_until + buffer_secs

    def check(self):
        self._load()
        now = time.time()

        # --- ban check ---
        ban_expires = self.banned_until + 900
        if now < ban_expires:
            raise RuntimeError(f"IP_BANNED — wait {int(ban_expires - now)}s")

        # --- rate limit check ---
        rate_limit_expires = self.rate_limited_until + 60
        if now < rate_limit_expires:
            wait = rate_limit_expires - now
            print(f"[RATE LIMITER] sleeping {wait:.1f}s before next request")
            time.sleep(wait)

        # --- weight guard: reset window if >60s old ---
        if now - self._weight_window_start > 60:
            self.current_weight = 0
            self._weight_window_start = now
            self._save()

        # --- preemptive throttle before Binance pulls the trigger ---
        if self.current_weight >= 1100:
            # hard block until the weight window resets
            window_remaining = 60 - (now - self._weight_window_start)
            wait = max(window_remaining, 1)
            print(f"[RATE LIMITER] weight={self.current_weight} >= 1100 — sleeping {wait:.1f}s for window reset")
            time.sleep(wait)
            # reset after sleeping
            self.current_weight = 0
            self._weight_window_start = time.time()
            self._save()
        elif self.current_weight >= 900:
            # progressive back-off: scale 0→10s as weight goes 900→1100
            wait = (self.current_weight - 900) / 200 * 10
            print(f"[RATE LIMITER] weight={self.current_weight} — throttling {wait:.1f}s")
            time.sleep(wait)

    def on_response(self, used_weight: int):
        """Call this after every successful Binance response."""
        self._load()
        now = time.time()

        # reset window if stale
        if now - self._weight_window_start > 60:
            self.current_weight = 0
            self._weight_window_start = now

        self.current_weight = max(self.current_weight, used_weight)
        self._save()

        # log when climbing into danger zone
        if used_weight >= 900:
            print(f"[RATE LIMITER] ⚠️  weight={used_weight} — approaching limit")

    def on_429(self, retry_after=None):
        self._load()
        self.rate_limited_until = time.time() + (retry_after or 60)
        self.current_weight = 1200  # assume maxed
        print(f"[RATE LIMITER] 429 — blocking all requests for {retry_after or 60}s")
        self._save()

    def on_418(self, retry_after=None):
        self._load()
        self.banned_until = time.time() + (retry_after or 7200)
        self.current_weight = 1200
        print(f"[RATE LIMITER] 418 — IP banned, blocking all requests for {retry_after or 7200}s")
        self._save()

rate_limiter = BinanceRateLimiter()