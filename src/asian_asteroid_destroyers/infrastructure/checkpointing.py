import sys
from pathlib import Path
from types import FrameType
from typing import Callable

from asian_asteroid_destroyers.genetic_algorithm import TrainingResult


class EmergencySaver:
    output_directory: Path
    _last_known_state: TrainingResult | None
    _last_known_generation: int

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self._last_known_state = None
        self._last_known_generation = 0

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

        print("Emergency checkpoint sequence completed. Exiting.")
        sys.exit(0)


def make_checkpoint_saver(
    output_directory: str | Path,
    save_interval: int = 10,
) -> Callable[[int, TrainingResult], None]:
    path = Path(output_directory)
    path.mkdir(parents=True, exist_ok=True)

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

    return saver_callback
