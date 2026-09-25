"""Overload protection: per-IP rate limits and the inference thread pool."""

import asyncio
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import time

from fastapi import HTTPException, Request, WebSocket
import torch

from src.production import settings


def client_ip(conn: Request | WebSocket) -> str:
    # Caddy is the only proxy in front of this container and appends the real
    # client address as the last X-Forwarded-For entry.
    forwarded = conn.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return conn.client.host if conn.client else "unknown"


class RateLimiter:
    """Sliding-window limiter per key (in memory, single process)."""

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self.hits = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if len(self.hits) > 10_000:
            self.hits = defaultdict(deque, {
                k: v for k, v in self.hits.items() if v and now - v[-1] < self.window
            })
        q = self.hits[key]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


predict_limiter = RateLimiter(settings.PREDICT_RATE_LIMIT, settings.PREDICT_RATE_WINDOW)
review_limiter = RateLimiter(settings.REVIEW_RATE_LIMIT, settings.REVIEW_RATE_WINDOW)


def rate_limit(limiter: RateLimiter):
    """FastAPI dependency that answers 429 when the client is over the limit."""
    def dependency(request: Request):
        if not limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail="Too many requests", headers={"Retry-After": "60"})
    return dependency


class Busy(Exception):
    pass


def init_inference_thread():
    # Must run inside each worker thread: with OpenMP the thread count is a
    # per-thread setting, so setting it once in the main thread is not enough.
    torch.set_num_threads(settings.TORCH_THREADS)


class InferenceGate:
    """Runs predictions on `parallel` worker threads. At most
    `parallel + queued` jobs (running + waiting) are admitted; everything
    else is rejected at once instead of queueing up and eating memory.

    The models are in eval mode under torch.inference_mode(), so several
    threads can safely run forward passes on the same model object."""

    def __init__(self, parallel: int, queued: int):
        self.max_pending = parallel + queued
        self.pending = 0
        self.executor = ThreadPoolExecutor(
            max_workers=parallel,
            thread_name_prefix="inference",
            initializer=init_inference_thread,
        )

    async def run(self, fn, *args):
        if self.pending >= self.max_pending:
            raise Busy
        self.pending += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)
        finally:
            self.pending -= 1


gate = InferenceGate(settings.PARALLEL_INFERENCES, settings.QUEUED_INFERENCES)
