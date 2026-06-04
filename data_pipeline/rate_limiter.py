# data_pipeline/rate_limiter.py
import time
import json
import os

STATE_FILE = "data/rate_limiter_state.json"

class BinanceRateLimiter:
    def __init__(self):
        self.banned_until = 0
        self.rate_limited_until = 0
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.banned_until = state.get("banned_until", 0)
                self.rate_limited_until = state.get("rate_limited_until", 0)
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE + ".tmp", "w") as f:
            json.dump({
                "banned_until": self.banned_until,
                "rate_limited_until": self.rate_limited_until,
            }, f)
        os.replace(STATE_FILE + ".tmp", STATE_FILE)

    def is_banned(self, buffer_secs=300) -> bool:
        self._load()
        return time.time() < self.banned_until + buffer_secs

    def check(self):
        self._load()
        now = time.time()
        ban_expires = self.banned_until + 300
        if now < ban_expires:
            raise RuntimeError(f"IP_BANNED — wait {int(ban_expires - now)}s")
        if now < self.rate_limited_until:
            wait = self.rate_limited_until - now
            print(f"[RATE LIMITER] sleeping {wait:.1f}s before next request")
            time.sleep(wait)

    def on_429(self, retry_after=None):
        self.rate_limited_until = time.time() + (retry_after or 60)
        print(f"[RATE LIMITER] 429 — blocking all requests for {retry_after or 60}s")
        self._save()

    def on_418(self, retry_after=None):
        self.banned_until = time.time() + (retry_after or 7200)
        print(f"[RATE LIMITER] 418 — IP banned, blocking all requests for {retry_after or 7200}s")
        self._save()

rate_limiter = BinanceRateLimiter()