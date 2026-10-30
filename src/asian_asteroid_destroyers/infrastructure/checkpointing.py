import datetime
import gzip
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

EMERGENCY_RUNTIME_FILENAME = "emergency_runtime.pkl.gz"
EMERGENCY_LOGBOOK_FILENAME = "emergency_logbook.parquet"
EMERGENCY_HISTORY_FILENAME = "emergency_history.parquet"

PERIODIC_RUNTIME_PATTERN = "runtime_gen_*.pkl.gz"


def get_periodic_runtime_filename(generation: int) -> str:
    return f"runtime_gen_{generation}.pkl.gz"


def get_periodic_logbook_filename(generation: int) -> str:
    return f"logbook_gen_{generation}.parquet"


def get_periodic_history_filename(generation: int) -> str:
    return f"history_gen_{generation}.parquet"


class CheckpointPayload(TypedDict):
    algorithm_path: str
    generation: int
    state: TrainingResult
    rng_state: Mapping[str, object]


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
            log_path = self.output_directory / EMERGENCY_LOGBOOK_FILENAME
            last_known_logbook.save_parquet(log_path)
            print(f"   -> Metrics logbook saved to: {log_path}")

        last_known_history_logger = self._last_known_state.history_logger
        if last_known_history_logger is not None:
            history_path = self.output_directory / EMERGENCY_HISTORY_FILENAME
            last_known_history_logger.save_parquet(history_path)
            print(f"   -> Hall of fame history saved to: {history_path}")

        runtime_path = self.output_directory / EMERGENCY_RUNTIME_FILENAME
        payload_data: CheckpointPayload = {
            "algorithm_path": self._algorithm_path,
            "generation": self._last_known_generation,
            "state": _create_lightweight_state(self._last_known_state),
            "rng_state": self._rng.bit_generator.state,
        }
        with gzip.open(runtime_path, "wb") as file:
            pickle.dump(payload_data, file)
        print(
            "   -> Compressed complete runtime state secured to: "
            + f"{runtime_path}"
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
                    path / get_periodic_logbook_filename(generation)
                )

            current_history_logger = current_state.history_logger
            if current_history_logger is not None:
                current_history_logger.save_parquet(
                    path / get_periodic_history_filename(generation)
                )

            runtime_path = path / get_periodic_runtime_filename(generation)
            payload_data: CheckpointPayload = {
                "algorithm_path": algorithm_path,
                "generation": generation,
                "state": _create_lightweight_state(current_state),
                "rng_state": rng.bit_generator.state,
            }
            with gzip.open(runtime_path, "wb") as file:
                pickle.dump(payload_data, file)

    return saver_callback


def _create_lightweight_state(state: TrainingResult) -> TrainingResult:
    return TrainingResult(
        final_population=state.final_population,
        logbook=None,
        history_logger=None,
        hall_of_fame=state.hall_of_fame,
    )


def create_experiment_directory(base_directory: str | Path = "./data") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = Path(base_directory) / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory
