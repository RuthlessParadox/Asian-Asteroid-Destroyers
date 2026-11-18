from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from asian_asteroid_destroyers.config import (
    EvaluationArgs,
    get_evaluation_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_get_evaluation_config_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "history.parquet"
    logbook_file = tmp_path / "logbook.parquet"
    scenarios_file = tmp_path / "scenarios.json"

    _ = history_file.write_text("dummy history content")
    _ = logbook_file.write_text("dummy logbook content")
    _ = scenarios_file.write_text("dummy scenarios content")

    test_args = [
        "evaluation.py",
        "--history-path",
        str(history_file),
        "--logbook-path",
        str(logbook_file),
        "--scenarios-path",
        str(scenarios_file),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    config = get_evaluation_config()

    assert config == EvaluationArgs(
        history_path=history_file,
        logbook_path=logbook_file,
        scenarios_path=scenarios_file,
    )


def test_get_evaluation_config_missing_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluation.py"])

    with pytest.raises(SystemExit):
        _ = get_evaluation_config()


def test_get_evaluation_config_file_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logbook_file = tmp_path / "logbook.parquet"
    scenarios_file = tmp_path / "scenarios.json"
    non_existent_history = tmp_path / "non_existent.parquet"

    _ = logbook_file.write_text("dummy")
    _ = scenarios_file.write_text("dummy")

    test_args = [
        "evaluation.py",
        "--history-path",
        str(non_existent_history),
        "--logbook-path",
        str(logbook_file),
        "--scenarios-path",
        str(scenarios_file),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(FileNotFoundError, match="History file not found"):
        _ = get_evaluation_config()
