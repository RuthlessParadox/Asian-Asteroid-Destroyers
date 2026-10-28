from pathlib import Path

import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    HistoryLogger,
    Logbook,
    TrainingResult,
)
from asian_asteroid_destroyers.infrastructure import make_checkpoint_saver


@pytest.fixture
def sample_logbook() -> Logbook:
    logbook = Logbook(flush_interval=1)
    logbook.record(
        generation=10,
        stats={
            "max_fitness": 1.0,
            "min_fitness": 0.0,
            "mean_fitness": 0.5,
            "median_fitness": 0.5,
            "std_fitness": 0.1,
        },
    )
    return logbook


@pytest.fixture
def sample_history() -> HistoryLogger:
    return HistoryLogger(flush_interval=1)


def test_make_checkpoint_saver_creates_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "checkpoints"
    _ = make_checkpoint_saver(output_dir)

    assert output_dir.is_dir()


def test_saver_callback_saves_on_matching_interval(
    tmp_path: Path,
    sample_logbook: Logbook,
    sample_history: HistoryLogger,
) -> None:
    output_dir = tmp_path / "checkpoints"
    checkpoint_saver = make_checkpoint_saver(output_dir, save_interval=10)

    state = TrainingResult(
        final_population=[],
        logbook=sample_logbook,
        history_logger=sample_history,
        hall_of_fame=None,
    )
    checkpoint_saver(10, state)

    expected_logbook = output_dir / "logbook_gen_10.parquet"
    expected_history = output_dir / "history_gen_10.parquet"

    assert expected_logbook.is_file()
    assert expected_history.is_file()


def test_saver_callback_skips_on_non_matching_interval(
    tmp_path: Path,
    sample_logbook: Logbook,
    sample_history: HistoryLogger,
) -> None:
    output_dir = tmp_path / "checkpoints"
    checkpoint_saver = make_checkpoint_saver(output_dir, save_interval=10)

    state = TrainingResult(
        final_population=[],
        logbook=sample_logbook,
        history_logger=sample_history,
        hall_of_fame=None,
    )
    checkpoint_saver(5, state)

    written_files = list(output_dir.iterdir())
    assert len(written_files) == 0


def test_saver_callback_handles_none_components(tmp_path: Path) -> None:
    output_dir = tmp_path / "checkpoints"
    checkpoint_saver = make_checkpoint_saver(output_dir, save_interval=1)

    state = TrainingResult(
        final_population=[],
        logbook=None,
        history_logger=None,
        hall_of_fame=None,
    )
    checkpoint_saver(1, state)

    written_files = [f for f in output_dir.iterdir() if f.is_file()]
    assert len(written_files) == 0
