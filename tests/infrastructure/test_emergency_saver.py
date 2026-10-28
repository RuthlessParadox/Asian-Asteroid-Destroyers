import inspect
from pathlib import Path

import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    HistoryLogger,
    Logbook,
    TrainingResult,
)
from asian_asteroid_destroyers.infrastructure import EmergencySaver


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "emergency_saves"


@pytest.fixture
def sample_logbook() -> Logbook:
    return Logbook(flush_interval=10)


@pytest.fixture
def sample_history_logger() -> HistoryLogger:
    return HistoryLogger(flush_interval=10)


def test_emergency_saver_initialization(temp_output_dir: Path) -> None:
    emergency_saver = EmergencySaver(temp_output_dir)
    assert emergency_saver.output_directory == temp_output_dir


def test_handle_interrupt_exits_when_no_state_available(
    temp_output_dir: Path,
) -> None:
    emergency_saver = EmergencySaver(temp_output_dir)

    with pytest.raises(SystemExit) as exception_info:
        emergency_saver.handle_interrupt(2, inspect.currentframe())

    assert exception_info.value.code == 1


def test_handle_interrupt_saves_checkpoint_and_exits_successfully(
    temp_output_dir: Path,
    sample_logbook: Logbook,
    sample_history_logger: HistoryLogger,
) -> None:
    emergency_saver = EmergencySaver(temp_output_dir)

    state = TrainingResult(
        final_population=[],
        logbook=sample_logbook,
        history_logger=sample_history_logger,
        hall_of_fame=None,
    )

    emergency_saver.update_state(generation=42, state=state)

    with pytest.raises(SystemExit) as exception_info:
        emergency_saver.handle_interrupt(2, inspect.currentframe())

    assert exception_info.value.code == 0

    expected_logbook_path = temp_output_dir / "emergency_logbook.parquet"
    expected_history_path = temp_output_dir / "emergency_history.parquet"

    assert expected_logbook_path.is_file()
    assert expected_history_path.is_file()


def test_handle_interrupt_handles_missing_loggers_gracefully(
    temp_output_dir: Path,
) -> None:
    emergency_saver = EmergencySaver(temp_output_dir)

    state = TrainingResult(
        final_population=[],
        logbook=None,
        history_logger=None,
        hall_of_fame=None,
    )

    emergency_saver.update_state(generation=10, state=state)

    with pytest.raises(SystemExit) as exception_info:
        emergency_saver.handle_interrupt(15, inspect.currentframe())

    assert exception_info.value.code == 0
    assert temp_output_dir.exists()
