import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable
import functools

class RateLimiter:
    """
    Sliding window rate limiter for requests per minute.
    Used for both source (read) and target (push) rate limiting.
    """
    def __init__(self, rate_per_minute: int):
        self.rate_per_minute = rate_per_minute
        self.window_seconds = 60
        self.request_times: list[float] = []  # Timestamps of requests in current window
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary. Logs time spent waiting."""
        import time
        wait_total = 0.0
        while True:
            async with self.lock:
                now = datetime.now().timestamp()
                cutoff = now - self.window_seconds
                self.request_times = [t for t in self.request_times if t > cutoff]
                if len(self.request_times) < self.rate_per_minute:
                    self.request_times.append(now)
                    if wait_total > 0.01:
                        import logging
                        logging.getLogger(__name__).info(f"RateLimiter waited {wait_total:.3f}s before allowing request")
                    return
                oldest = self.request_times[0]
                wait_time = (oldest + self.window_seconds) - now
            if wait_time > 0:
                await asyncio.sleep(wait_time + 0.01)
                wait_total += wait_time + 0.01
            else:
                await asyncio.sleep(0.01)
                wait_total += 0.01

def read_rate_limited(async_func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """
    Decorator for async (read) functions to apply a source (read) RateLimiter if 'source_rate_limiter' is passed as a kwarg.
    If present, acquires the source (read) rate limiter before calling the function.
    Used for limiting GET/read requests to the source STAC API.
    """
    @functools.wraps(async_func)
    async def wrapper(*args, **kwargs) -> Any:
        source_rate_limiter = kwargs.get('source_rate_limiter')
        if source_rate_limiter is not None:
            await source_rate_limiter.acquire()
        return await async_func(*args, **kwargs)
    return wrapper
