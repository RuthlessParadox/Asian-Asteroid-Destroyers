import datetime
from pathlib import Path
from types import ModuleType

import pytest
from pytest import MonkeyPatch

from asian_asteroid_destroyers.infrastructure import (
    create_experiment_directory,
)


@pytest.fixture
def mock_now(monkeypatch: MonkeyPatch) -> datetime.datetime:
    fixed_time = datetime.datetime(
        year=2026,
        month=10,
        day=28,
        hour=10,
        minute=15,
        second=30,
    )

    mock_datetime_module = ModuleType("datetime")

    def mock_now_factory(
        _timezone: datetime.tzinfo | None = None,
    ) -> datetime.datetime:
        return fixed_time

    mock_datetime_class = type(
        "datetime",
        (),
        {"now": staticmethod(mock_now_factory)},
    )
    setattr(mock_datetime_module, "datetime", mock_datetime_class)

    monkeypatch.setattr(
        "asian_asteroid_destroyers.infrastructure.checkpointing.datetime",
        mock_datetime_module,
    )
    return fixed_time


def test_create_experiment_directory_default_path(
    mock_now: datetime.datetime,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    expected_timestamp = mock_now.strftime("%Y%m%d_%H%M%S")
    expected_directory = tmp_path / "data" / f"run_{expected_timestamp}"

    actual_directory = create_experiment_directory().resolve()

    assert actual_directory == expected_directory
    assert actual_directory.is_dir()


def test_create_experiment_directory_custom_path(
    mock_now: datetime.datetime,
    tmp_path: Path,
) -> None:
    expected_timestamp = mock_now.strftime("%Y%m%d_%H%M%S")
    expected_directory = tmp_path / f"run_{expected_timestamp}"

    actual_directory = create_experiment_directory(base_directory=tmp_path)

    assert actual_directory == expected_directory
    assert actual_directory.is_dir()


def test_create_experiment_directory_handles_existing(
    mock_now: datetime.datetime,
    tmp_path: Path,
) -> None:
    expected_timestamp = mock_now.strftime("%Y%m%d_%H%M%S")
    expected_directory = tmp_path / f"run_{expected_timestamp}"

    expected_directory.mkdir(parents=True, exist_ok=True)

    actual_directory = create_experiment_directory(base_directory=tmp_path)

    assert actual_directory == expected_directory
    assert actual_directory.is_dir()
