# Generations of best fitness
# Number of generations till target fitness
# Number of generations without improvement
# Final best and average fitness
# Percentage of runs that coverage

# Training Runtime
# Generation Runtime
# Evaluation Runtime
# Individuals Evaluated Per Second
# CPU and Memory Usage
# Best, Average, Median Fitness
# Stagnation Count
# Scenario Gameplay Metrics

# should only collect data; not control training

# TODO field to store time;  private vars; fields; class should know the time; programmers shouldn't be responsible
# TODO add onto data array and then flush for every amount of entries,
# TODO accumulate into dataframe, group dataframes together
# TODO save data to disc

from __future__ import annotations

import statistics
import time

import polars as pl


class BenchMarkMetrics:
    def __init__(self, flush_interval: int = 100):

        self.training_start: float | None = None
        self.generation_start: float | None = None
        self.evaluation_start: float | None = None
        self.generation_records: list[dict[str, float]] = []
        self.generation_chunks: list[pl.DataFrame] = []

        # Runtime
        self._training_runtime = 0.0
        self.generation_times: list[float] = []
        self.evaluation_times: list[float] = []

        # fitness measurements
        self.best_fitness: list[float] = []
        self.average_fitness: list[float] = []
        self.median_fitness: list[float] = []
        self.fitness_std: list[float] = []

        # data storage
        self.buffer: list[dict[str, float]] = []
        self.dataframes: list[pl.DataFrame] = []
        self.flush_interval = flush_interval

    def flush(self) -> None:
        if not self.generation_records:
            return
        chunk = pl.DataFrame(self.generation_records)
        self.generation_chunks.append(chunk)
        self.generation_records.clear()

    @property
    def dataframe(self) -> pl.DataFrame:
        self.flush()
        if not self.generation_chunks:
            return pl.DataFrame()
        return pl.concat(self.generation_chunks)

    def start_training(self) -> None:
        self.training_start = time.perf_counter()

    def finish_training(self) -> None:
        if self.training_start is None:
            return
        self._training_runtime = time.perf_counter() - self.training_start
        self.training_start = None

    @property
    def training_runtime(self) -> float:
        if self.training_start is not None:
            return time.perf_counter() - self.training_start
        return self._training_runtime

    def start_generation(self) -> None:
        self.generation_start = time.perf_counter()

    def finish_generation(
        self, generation: int, fitness_values: list[float]
    ) -> None:
        if self.generation_start is None:
            return
        elapsed = time.perf_counter() - self.generation_start
        self.generation_start = None
        if not fitness_values:
            return
        best = max(fitness_values)
        average = statistics.mean(fitness_values)
        median = statistics.median(fitness_values)

        std = (
            statistics.stdev(fitness_values)
            if len(fitness_values) > 1
            else 0.0
        )
        self.generation_times.append(elapsed)
        self.best_fitness.append(best)
        self.average_fitness.append(average)
        self.median_fitness.append(median)
        self.fitness_std.append(std)
        self.buffer.append(
            {
                "generation": generation,
                "generation_runtime": elapsed,
                "best_fitness": best,
                "average_fitness": average,
                "median_fitness": median,
                "fitness_std": std,
            }
        )
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def record_generation(
        self, generation: int, fitness_values: list[float]
    ) -> None:
        if self.generation_start is None:
            raise RuntimeError()
        if not fitness_values:
            return
        elapsed = time.perf_counter() - self.generation_start
        self.generation_records.append(
            {
                "generation": generation,
                "generation_runtime": elapsed,
                "best_fitness": max(fitness_values),
                "average_fitness": statistics.mean(fitness_values),
                "median_fitness": statistics.median(fitness_values),
                "fitness_std": (
                    statistics.stdev(fitness_values)
                    if len(fitness_values) > 1
                    else 0.0
                ),
            }
        )
        self.generation_start = None
        if len(self.generation_records) >= self.flush_interval:
            self.flush()

    def start_evaluation(self) -> None:
        self.evaluation_start = time.perf_counter()

    def finish_evaluation(self) -> float:
        if self.evaluation_start is None:
            raise RuntimeError()
        elapsed = time.perf_counter() - self.evaluation_start
        self.evaluation_start = None
        return elapsed

    def save_parquet(self, path: str) -> None:
        self.dataframe.write_parquet(
            path,
            compression="zstd",
            compression_level=3,
        )
