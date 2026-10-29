import datetime
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from types import FrameType
from typing import Callable, TypedDict

import numpy as np

from asian_asteroid_destroyers.genetic_algorithm import (
    EvolutionAlgorithm,
    TrainingResult,
)


class CheckpointPayload(TypedDict):
    generation: int
    state: TrainingResult
    rng_state: Mapping[str, object]
    algorithm_path: str


class EmergencySaver:
    output_directory: Path
    _last_known_state: TrainingResult | None
    _last_known_generation: int
    _rng: np.random.Generator
    _algorithm_path: str

    def __init__(
        self,
        output_directory: str | Path,
        rng: np.random.Generator,
        algorithm: EvolutionAlgorithm[TrainingResult],
    ) -> None:
        self.output_directory = Path(output_directory)
        self._last_known_state = None
        self._last_known_generation = 0
        self._rng = rng
        self._algorithm_path = f"{algorithm.__module__}.{algorithm.__name__}"

    def update_state(self, generation: int, state: TrainingResult) -> None:
        self._last_known_generation = generation
        self._last_known_state = state

    def handle_interrupt(self, signum: int, _frame: FrameType | None) -> None:
        print(
            f"\nSignal {signum} (Ctrl+C) captured. "
            + "Executing emergency save sequence..."
        )

        if self._last_known_state is None:
            print("No generation checkpoints available yet. Exiting.")
            sys.exit(1)

        self.output_directory.mkdir(parents=True, exist_ok=True)
        print(
            f"Securing data from generation {self._last_known_generation}..."
        )

        last_known_logbook = self._last_known_state.logbook
        if last_known_logbook is not None:
            log_path = self.output_directory / "emergency_logbook.parquet"
            last_known_logbook.save_parquet(log_path)
            print(f"   -> Metrics logbook saved to: {log_path}")

        last_known_history_logger = self._last_known_state.history_logger
        if last_known_history_logger is not None:
            history_path = self.output_directory / "emergency_history.parquet"
            last_known_history_logger.save_parquet(history_path)
            print(f"   -> Hall of fame history saved to: {history_path}")

        runtime_path = self.output_directory / "emergency_runtime.pkl"
        payload_data: CheckpointPayload = {
            "generation": self._last_known_generation,
            "state": self._last_known_state,
            "rng_state": self._rng.bit_generator.state,
            "algorithm_path": self._algorithm_path,
        }
        with open(runtime_path, "wb") as file:
            pickle.dump(payload_data, file)
        print(
            f"   -> Complete runtime state pickling secured to: {runtime_path}"
        )

        print("Emergency checkpoint sequence completed. Exiting.")
        sys.exit(0)


def make_checkpoint_saver(
    output_directory: str | Path,
    rng: np.random.Generator,
    algorithm: EvolutionAlgorithm[TrainingResult],
    save_interval: int = 10,
) -> Callable[[int, TrainingResult], None]:
    path = Path(output_directory)
    path.mkdir(parents=True, exist_ok=True)
    algorithm_path = f"{algorithm.__module__}.{algorithm.__name__}"

    def saver_callback(generation: int, current_state: TrainingResult) -> None:
        if generation % save_interval == 0:
            print(f"Saving periodic checkpoint for generation {generation}...")

            current_logbook = current_state.logbook
            if current_logbook is not None:
                current_logbook.save_parquet(
                    path / f"logbook_gen_{generation}.parquet"
                )

            current_history_logger = current_state.history_logger
            if current_history_logger is not None:
                current_history_logger.save_parquet(
                    path / f"history_gen_{generation}.parquet"
                )

            runtime_path = path / f"runtime_gen_{generation}.pkl"
            payload_data: CheckpointPayload = {
                "generation": generation,
                "state": current_state,
                "rng_state": rng.bit_generator.state,
                "algorithm_path": algorithm_path,
            }
            with open(runtime_path, "wb") as file:
                pickle.dump(payload_data, file)

    return saver_callback


def create_experiment_directory(base_directory: str | Path = "./data") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = Path(base_directory) / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory
