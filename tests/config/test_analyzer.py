from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from asian_asteroid_destroyers.config import get_analyzer_config

if TYPE_CHECKING:
    from pathlib import Path


def test_get_analyzer_config_minimal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyzer.py", "--run-dir", str(tmp_path)],
    )

    config = get_analyzer_config()

    assert config.run_dir == tmp_path
    assert config.track_genes == []


def test_get_analyzer_config_with_genes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyzer.py",
            "--run-dir",
            str(tmp_path),
            "--track-genes",
            "10",
            "20",
            "30",
        ],
    )

    config = get_analyzer_config()

    assert config.run_dir == tmp_path
    assert config.track_genes == [10, 20, 30]


def test_get_analyzer_config_missing_run_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["analyzer.py"])

    with pytest.raises(SystemExit):
        _ = get_analyzer_config()


def test_get_analyzer_config_dir_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    non_existent = tmp_path / "non_existent_workspace"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyzer.py", "--run-dir", str(non_existent)],
    )

    with pytest.raises(
        FileNotFoundError,
        match="Target workspace path does not exist",
    ):
        _ = get_analyzer_config()


def test_get_analyzer_config_invalid_track_genes_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyzer.py",
            "--run-dir",
            str(tmp_path),
            "--track-genes",
            "not-an-int",
        ],
    )

    with pytest.raises(SystemExit):
        _ = get_analyzer_config()


def test_get_analyzer_config_mixed_valid_and_invalid_genes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyzer.py",
            "--run-dir",
            str(tmp_path),
            "--track-genes",
            "10",
            "abc",
            "30",
        ],
    )

    with pytest.raises(SystemExit):
        _ = get_analyzer_config()


def test_get_analyzer_config_negative_gene_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyzer.py", "--run-dir", str(tmp_path), "--track-genes", "-5"],
    )

    with pytest.raises(SystemExit):
        _ = get_analyzer_config()


def test_get_analyzer_config_empty_track_genes_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyzer.py", "--run-dir", str(tmp_path), "--track-genes"],
    )

    with pytest.raises(SystemExit):
        _ = get_analyzer_config()


def test_get_analyzer_config_duplicate_genes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyzer.py",
            "--run-dir",
            str(tmp_path),
            "--track-genes",
            "10",
            "20",
            "10",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate gene indices are not allowed",
    ):
        _ = get_analyzer_config()
