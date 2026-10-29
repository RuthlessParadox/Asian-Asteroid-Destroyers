import inspect
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    EvolutionAlgorithm,
    HistoryLogger,
    Logbook,
    TrainingResult,
)
from asian_asteroid_destroyers.infrastructure import (
    CheckpointPayload,
    EmergencySaver,
)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "emergency_saves"


@pytest.fixture
def sample_logbook() -> Logbook:
    return Logbook(flush_interval=10)


@pytest.fixture
def sample_history_logger() -> HistoryLogger:
    return HistoryLogger(flush_interval=10)


@pytest.fixture
def shared_rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def mock_algorithm() -> EvolutionAlgorithm[TrainingResult]:
    class MockAlgorithm:
        __name__: str = "mock_evolution_loop"
        __module__: str = "test_package.mock_module"

        def __call__(self, *args: object, **kwargs: object) -> TrainingResult:
            return TrainingResult(
                final_population=[],
                logbook=None,
                history_logger=None,
                hall_of_fame=None,
            )

    return MockAlgorithm()


def test_emergency_saver_initialization(
    temp_output_dir: Path,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    emergency_saver = EmergencySaver(
        output_directory=temp_output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
    )
    assert emergency_saver.output_directory == temp_output_dir


def test_handle_interrupt_exits_when_no_state_available(
    temp_output_dir: Path,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    emergency_saver = EmergencySaver(
        output_directory=temp_output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
    )

    with pytest.raises(SystemExit) as exception_info:
        emergency_saver.handle_interrupt(2, inspect.currentframe())

    assert exception_info.value.code == 1


def test_handle_interrupt_saves_checkpoint_and_exits_successfully(
    temp_output_dir: Path,
    sample_logbook: Logbook,
    sample_history_logger: HistoryLogger,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    emergency_saver = EmergencySaver(
        output_directory=temp_output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
    )

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
    expected_runtime_path = temp_output_dir / "emergency_runtime.pkl"

    assert expected_logbook_path.is_file()
    assert expected_history_path.is_file()
    assert expected_runtime_path.is_file()

    with open(expected_runtime_path, "rb") as file:
        payload = cast(CheckpointPayload, pickle.load(file))
        assert payload["generation"] == 42
        assert (
            payload["algorithm_path"]
            == "test_package.mock_module.mock_evolution_loop"
        )
        assert payload["rng_state"] == shared_rng.bit_generator.state


def test_handle_interrupt_handles_missing_loggers_gracefully(
    temp_output_dir: Path,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    emergency_saver = EmergencySaver(
        output_directory=temp_output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
    )

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

    expected_runtime_path = temp_output_dir / "emergency_runtime.pkl"
    assert expected_runtime_path.is_file()
