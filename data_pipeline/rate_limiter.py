# data_pipeline/rate_limiter.py
import time

class BinanceRateLimiter:
    def __init__(self):
        self.banned_until = 0
        self.rate_limited_until = 0

    def check(self):
        now = time.time()
        if now < self.banned_until:
            raise RuntimeError(f"IP_BANNED — wait {int(self.banned_until - now)}s")
        if now < self.rate_limited_until:
            wait = self.rate_limited_until - now
            print(f"[RATE LIMITER] sleeping {wait:.1f}s before next request")
            time.sleep(wait)

    def on_429(self, retry_after=None):
        self.rate_limited_until = time.time() + (retry_after or 60)
        print(f"[RATE LIMITER] 429 — blocking all requests for {retry_after or 60}s")

    def on_418(self, retry_after=None):
        self.banned_until = time.time() + (retry_after or 7200)
        print(f"[RATE LIMITER] 418 — IP banned, blocking all requests for {retry_after or 7200}s")

rate_limiter = BinanceRateLimiter()