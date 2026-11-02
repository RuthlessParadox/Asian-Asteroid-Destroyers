import datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from pytest import MonkeyPatch

from asian_asteroid_destroyers.infrastructure import (
    POINTER_FILENAME,
    create_experiment_directory,
    update_active_pointer,
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
        "asian_asteroid_destroyers.infrastructure.workspace.datetime",
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


def test_update_active_pointer_writes_atomically(tmp_path: Path) -> None:
    checkpoint_name = "runtime_gen_10.pkl.gz"
    pointer_file = tmp_path / POINTER_FILENAME
    temp_pointer_file = pointer_file.with_suffix(".tmp")

    update_active_pointer(tmp_path, checkpoint_name)

    assert pointer_file.is_file()
    assert not temp_pointer_file.exists()
    assert pointer_file.read_text(encoding="utf-8").strip() == checkpoint_name


def test_update_active_pointer_overwrites_existing_pointers(
    tmp_path: Path,
) -> None:
    pointer_file = tmp_path / POINTER_FILENAME
    _ = pointer_file.write_text("runtime_gen_10.pkl.gz", encoding="utf-8")

    new_checkpoint_name = "runtime_gen_20.pkl.gz"

    update_active_pointer(tmp_path, new_checkpoint_name)

    assert (
        pointer_file.read_text(encoding="utf-8").strip() == new_checkpoint_name
    )


def test_update_active_pointer_isolates_filesystem_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_name = "runtime_gen_10.pkl.gz"

    with patch.object(
        Path,
        "write_text",
        side_effect=OSError("Permission Denied"),
    ):
        update_active_pointer(tmp_path, checkpoint_name)

    captured = capsys.readouterr()
    assert (
        "Warning: Could not rotate active checkpoint pointer: Permission Denied"
        in captured.out
    )

    pointer_file = tmp_path / POINTER_FILENAME
    assert not pointer_file.exists()
