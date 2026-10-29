import pickle
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    EvolutionAlgorithm,
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


@pytest.fixture
def shared_rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def mock_algorithm() -> EvolutionAlgorithm[TrainingResult]:
    class MockAlgorithm:
        __name__: str = "periodic_evolution_loop"
        __module__: str = "test_package.mock_module"

        def __call__(self, *args: object, **kwargs: object) -> TrainingResult:
            return TrainingResult(
                final_population=[],
                logbook=None,
                history_logger=None,
                hall_of_fame=None,
            )

    return MockAlgorithm()


class _CheckpointPayload(TypedDict):
    generation: int
    state: TrainingResult
    rng_state: dict[str, object]
    algorithm_path: str


def test_make_checkpoint_saver_creates_directory(
    tmp_path: Path,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    output_dir = tmp_path / "checkpoints"
    _ = make_checkpoint_saver(
        output_directory=output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
    )

    assert output_dir.is_dir()


def test_saver_callback_saves_on_matching_interval(
    tmp_path: Path,
    sample_logbook: Logbook,
    sample_history: HistoryLogger,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    output_dir = tmp_path / "checkpoints"
    checkpoint_saver = make_checkpoint_saver(
        output_directory=output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
        save_interval=10,
    )

    state = TrainingResult(
        final_population=[],
        logbook=sample_logbook,
        history_logger=sample_history,
        hall_of_fame=None,
    )
    checkpoint_saver(10, state)

    expected_logbook = output_dir / "logbook_gen_10.parquet"
    expected_history = output_dir / "history_gen_10.parquet"
    expected_runtime = output_dir / "runtime_gen_10.pkl"

    assert expected_logbook.is_file()
    assert expected_history.is_file()
    assert expected_runtime.is_file()

    with open(expected_runtime, "rb") as file:
        payload = cast(_CheckpointPayload, pickle.load(file))
        assert payload["generation"] == 10
        assert (
            payload["algorithm_path"]
            == "test_package.mock_module.periodic_evolution_loop"
        )
        assert payload["rng_state"] == shared_rng.bit_generator.state


def test_saver_callback_skips_on_non_matching_interval(
    tmp_path: Path,
    sample_logbook: Logbook,
    sample_history: HistoryLogger,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    output_dir = tmp_path / "checkpoints"
    checkpoint_saver = make_checkpoint_saver(
        output_directory=output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
        save_interval=10,
    )

    state = TrainingResult(
        final_population=[],
        logbook=sample_logbook,
        history_logger=sample_history,
        hall_of_fame=None,
    )
    checkpoint_saver(5, state)

    written_files = list(output_dir.iterdir())
    assert len(written_files) == 0


def test_saver_callback_handles_none_components(
    tmp_path: Path,
    shared_rng: np.random.Generator,
    mock_algorithm: EvolutionAlgorithm[TrainingResult],
) -> None:
    output_dir = tmp_path / "checkpoints"
    checkpoint_saver = make_checkpoint_saver(
        output_directory=output_dir,
        rng=shared_rng,
        algorithm=mock_algorithm,
        save_interval=1,
    )

    state = TrainingResult(
        final_population=[],
        logbook=None,
        history_logger=None,
        hall_of_fame=None,
    )
    checkpoint_saver(1, state)

    expected_runtime = output_dir / "runtime_gen_1.pkl"
    assert expected_runtime.is_file()

    written_files = [file for file in output_dir.iterdir() if file.is_file()]
    assert len(written_files) == 1
