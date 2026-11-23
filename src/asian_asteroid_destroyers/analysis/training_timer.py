from __future__ import annotations

import statistics
import time

import polars as pl


def start_training(self) -> None:
    self.training_start = time.perf_counter()


def finish_training(self) -> None:
    if self.training_start is None:
        return
    self._training_runtime = time.perf_counter() - self.training_start
    self.training_start = None
