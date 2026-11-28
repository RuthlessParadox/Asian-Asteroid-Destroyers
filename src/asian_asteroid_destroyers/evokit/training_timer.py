from __future__ import annotations

import statistics
import time

import polars as pl


# needs to be a class
class TrainingTimer:
    __slots__: tuple[str, ...] = ("_start_time",)
    _start_time: float | None

    def __init__(self) -> None:
        self._start_time = None

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()

    def stop_timer(self) -> float:
        if self._start_time is None:
            raise ValueError("timer has not been started")

        return time.perf_counter() - self._start_time
