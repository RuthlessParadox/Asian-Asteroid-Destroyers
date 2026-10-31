import gzip
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from types import FrameType
from typing import Callable, TypedDict

import numpy as np

from asian_asteroid_destroyers.evokit import (
    EvolutionAlgorithm,
    HallOfFame,
    Population,
    TrainingResult,
)

from .parsing import extract_generation_number
from .workspace import (
    EMERGENCY_RUNTIME_FILENAME,
    PERIODIC_RUNTIME_PATTERN,
    get_periodic_runtime_filename,
    update_active_pointer,
)

EMERGENCY_LOGBOOK_FILENAME = "emergency_logbook.parquet"
EMERGENCY_HISTORY_FILENAME = "emergency_history.parquet"


def get_periodic_logbook_filename(generation: int) -> str:
    return f"logbook_gen_{generation}.parquet"


def get_periodic_history_filename(generation: int) -> str:
    return f"history_gen_{generation}.parquet"


class CheckpointPayload(TypedDict):
    algorithm_path: str
    generation: int
    final_population: Population
    hall_of_fame: HallOfFame | None
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
            "final_population": self._last_known_state.final_population,
            "hall_of_fame": self._last_known_state.hall_of_fame,
            "rng_state": self._rng.bit_generator.state,
        }
        with gzip.open(runtime_path, "wb") as file:
            pickle.dump(payload_data, file)
        print(
            "   -> Compressed complete runtime state secured to: "
            + f"{runtime_path}"
        )

        update_active_pointer(
            self.output_directory,
            EMERGENCY_RUNTIME_FILENAME,
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
    milestone_interval = save_interval * 10

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

            runtime_name = get_periodic_runtime_filename(generation)
            runtime_path = path / runtime_name
            payload_data: CheckpointPayload = {
                "algorithm_path": algorithm_path,
                "generation": generation,
                "final_population": current_state.final_population,
                "hall_of_fame": current_state.hall_of_fame,
                "rng_state": rng.bit_generator.state,
            }
            with gzip.open(runtime_path, "wb") as file:
                pickle.dump(payload_data, file)

            update_active_pointer(path, runtime_name)

            current_runtime_files = list(path.glob(PERIODIC_RUNTIME_PATTERN))
            for file_path in current_runtime_files:
                try:
                    past_generation = extract_generation_number(file_path)
                except ValueError:
                    continue

                if past_generation % milestone_interval == 0:
                    continue
                if past_generation == generation:
                    continue
                if past_generation == generation - save_interval:
                    continue

                _safely_delete_generation_files(path, past_generation)

    return saver_callback


def _safely_delete_generation_files(directory: Path, generation: int) -> None:
    targets = [
        directory / get_periodic_runtime_filename(generation),
        directory / get_periodic_logbook_filename(generation),
        directory / get_periodic_history_filename(generation),
    ]
    for target in targets:
        try:
            target.unlink(missing_ok=True)
        except OSError as error:
            print(
                f"Warning: Could not prune intermediate asset {target.name}: "
                + f"{error}"
            )
