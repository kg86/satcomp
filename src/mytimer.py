from __future__ import annotations

import time


class Timer:
    def __init__(self) -> None:
        self.t = time.time()
        self.times = dict()

    def __call__(self) -> float:
        x = time.time() - self.t
        self.t = time.time()
        return x

    def record(self, name: str) -> None:
        self.times[name] = time.time() - self.t
        self.t = time.time()
