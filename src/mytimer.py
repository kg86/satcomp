"""Lightweight timing helper used by solvers and benchmarks."""

from __future__ import annotations

import time


class Timer:
    """Collect named timing measurements."""

    def __init__(self) -> None:
        self.t = time.time()
        self.times = dict()

    def __call__(self) -> float:
        x = time.time() - self.t
        self.t = time.time()
        return x

    def record(self, name: str) -> None:
        """Record elapsed time since last tick under `name` and reset the timer."""
        self.times[name] = time.time() - self.t
        self.t = time.time()
