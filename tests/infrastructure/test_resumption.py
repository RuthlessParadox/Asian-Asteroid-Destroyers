import gzip
import pickle
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    TrainingResult,
)
from asian_asteroid_destroyers.infrastructure import (
    EMERGENCY_RUNTIME_FILENAME,
    CheckpointPayload,
    get_periodic_runtime_filename,
    load_resumption_workspace,
)

_MOCK_MODULE_NAME = "mock_evolution"
_MOCK_ALGORITHM_NAME = "run_mock_evolution"
_MOCK_FULL_PATH = f"{_MOCK_MODULE_NAME}.{_MOCK_ALGORITHM_NAME}"


def dummy_algorithm_target(
    _toolbox: object,
    _cx_pb: float,
    _mut_pb: float,
    _generations: int,
    _population_size: int,
    **_kwargs: object,
) -> TrainingResult:
    return TrainingResult(
        final_population=[],
        logbook=None,
        history_logger=None,
        hall_of_fame=None,
    )


@pytest.fixture(autouse=True)
def setup_mock_module() -> Generator[None, None, None]:
    mock_module = ModuleType(_MOCK_MODULE_NAME)
    setattr(mock_module, _MOCK_ALGORITHM_NAME, dummy_algorithm_target)

    original_module = sys.modules.get(_MOCK_MODULE_NAME)
    sys.modules[_MOCK_MODULE_NAME] = mock_module

    yield

    if original_module is not None:
        sys.modules[_MOCK_MODULE_NAME] = original_module
    elif _MOCK_MODULE_NAME in sys.modules:
        del sys.modules[_MOCK_MODULE_NAME]


@pytest.fixture
def workspace_directory(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_payload() -> CheckpointPayload:
    training_state = TrainingResult(
        final_population=[],
        logbook=None,
        history_logger=None,
        hall_of_fame=None,
    )
    rng_state = np.random.default_rng(seed=42).bit_generator.state

    return {
        "algorithm_path": _MOCK_FULL_PATH,
        "generation": 40,
        "state": training_state,
        "rng_state": rng_state,
    }


def test_load_resumption_workspace_prefers_emergency_snapshot(
    workspace_directory: Path,
    sample_payload: CheckpointPayload,
) -> None:
    periodic_payload = sample_payload.copy()
    periodic_payload["generation"] = 20
    periodic_path = workspace_directory / get_periodic_runtime_filename(20)
    with gzip.open(periodic_path, "wb") as file:
        pickle.dump(periodic_payload, file)

    emergency_payload = sample_payload.copy()
    emergency_payload["generation"] = 42
    emergency_path = workspace_directory / EMERGENCY_RUNTIME_FILENAME
    with gzip.open(emergency_path, "wb") as file:
        pickle.dump(emergency_payload, file)

    workspace = load_resumption_workspace(workspace_directory)

    assert workspace.last_generation == 42
    assert isinstance(workspace.rng, np.random.Generator)
    assert workspace.saved_state.final_population == []

    assert workspace.saved_state.logbook is not None
    assert workspace.saved_state.history_logger is not None


def test_load_resumption_workspace_fallback_to_max_periodic(
    workspace_directory: Path,
    sample_payload: CheckpointPayload,
) -> None:
    generations = [10, 50, 30]
    for generation in generations:
        payload = sample_payload.copy()
        payload["generation"] = generation
        file_path = workspace_directory / get_periodic_runtime_filename(
            generation,
        )
        with gzip.open(file_path, "wb") as file:
            pickle.dump(payload, file)

    workspace = load_resumption_workspace(workspace_directory)

    assert workspace.last_generation == 50


def test_load_resumption_workspace_raises_file_not_found(
    workspace_directory: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError, match="No valid state snapshots detected"
    ):
        _ = load_resumption_workspace(workspace_directory)


def test_load_resumption_workspace_rng_state_restoration(
    workspace_directory: Path,
    sample_payload: CheckpointPayload,
) -> None:
    base_bit_gen = np.random.PCG64(seed=12345)
    sample_payload["rng_state"] = base_bit_gen.state

    emergency_path = workspace_directory / EMERGENCY_RUNTIME_FILENAME
    with gzip.open(emergency_path, "wb") as file:
        pickle.dump(sample_payload, file)

    workspace = load_resumption_workspace(workspace_directory)

    expected_rng = np.random.Generator(np.random.PCG64(seed=12345))
    assert workspace.rng.random() == expected_rng.random()


def test_load_resumption_workspace_invalid_algorithm_path_raises_import_error(
    workspace_directory: Path,
    sample_payload: CheckpointPayload,
) -> None:
    sample_payload["algorithm_path"] = "non_existent_module.some_function"

    emergency_path = workspace_directory / EMERGENCY_RUNTIME_FILENAME
    with gzip.open(emergency_path, "wb") as file:
        pickle.dump(sample_payload, file)

    with pytest.raises(
        ImportError,
        match="Failed to dynamically import the target evolution module",
    ):
        _ = load_resumption_workspace(workspace_directory)
