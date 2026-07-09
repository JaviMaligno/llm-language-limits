from __future__ import annotations
import threading
import time as _time
from typing import Callable

from .config import PROVIDER_MIN_INTERVAL


class RateLimiter:
    """Enforce a minimum interval between acquire() calls per key, across threads."""

    def __init__(self, min_intervals: dict, *,
                 clock: Callable[[], float] = _time.monotonic,
                 sleep: Callable[[float], None] = _time.sleep):
        self._min = dict(min_intervals)          # key -> min seconds between calls
        self._clock = clock
        self._sleep = sleep
        self._next_allowed: dict = {}            # key -> earliest next monotonic time
        self._lock = threading.Lock()

    def acquire(self, key) -> None:
        interval = self._min.get(key, 0.0)
        if interval <= 0:
            return
        while True:
            with self._lock:
                now = self._clock()
                earliest = self._next_allowed.get(key, 0.0)
                if now >= earliest:
                    # reserve this slot and the next
                    self._next_allowed[key] = max(now, earliest) + interval
                    return
                wait = earliest - now
            self._sleep(wait)


LIMITER = RateLimiter(PROVIDER_MIN_INTERVAL)


def throttle(provider) -> None:
    LIMITER.acquire(provider)
