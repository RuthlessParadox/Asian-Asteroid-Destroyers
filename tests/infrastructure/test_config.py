import pathlib
import sys

import pytest

from asian_asteroid_destroyers.infrastructure import RuntimeArgs, get_config


@pytest.fixture(autouse=True)
def mock_env(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["test_program"])
    return tmp_path


@pytest.fixture
def valid_toml_content() -> str:
    return """
    [tool.evolution]
    generations = 100
    population_size = 50
    save_interval = 10
    cx_pb = 0.8
    mut_pb = 0.2
    gene_size = 30
    """


def test_get_config_from_baseline_only(
    tmp_path: pathlib.Path,
    valid_toml_content: str,
) -> None:
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(valid_toml_content, encoding="utf-8")

    result = get_config()

    assert isinstance(result, RuntimeArgs)
    assert result.generations == 100
    assert result.population_size == 50
    assert result.save_interval == 10
    assert result.cx_pb == 0.8
    assert result.mut_pb == 0.2
    assert result.gene_size == 30


def test_get_config_overridden_by_cli_args(
    tmp_path: pathlib.Path,
    valid_toml_content: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(valid_toml_content, encoding="utf-8")

    custom_args = ["test_program", "--generations", "200", "--cx-pb", "0.65"]
    monkeypatch.setattr(sys, "argv", custom_args)

    result = get_config()

    assert result.generations == 200
    assert result.cx_pb == 0.65
    assert result.population_size == 50
    assert result.mut_pb == 0.2


def test_get_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        _ = get_config()


def test_get_config_missing_evolution_table(tmp_path: pathlib.Path) -> None:
    config_file = tmp_path / "pyproject.toml"
    invalid_content = "[tool.other_tool]\nvalue = 42"
    _ = config_file.write_text(invalid_content, encoding="utf-8")

    with pytest.raises(KeyError):
        _ = get_config()


@pytest.mark.parametrize(
    "invalid_toml",
    [
        """
        [tool.evolution]
        generations = 10.5
        population_size = 50
        save_interval = 10
        cx_pb = 0.8
        mut_pb = 0.2
        gene_size = 30
        """,
        """
        [tool.evolution]
        generations = 100
        population_size = 50
        save_interval = 10
        cx_pb = 1
        mut_pb = 0.2
        gene_size = 30
        """,
    ],
)
def test_get_config_invalid_types(
    tmp_path: pathlib.Path,
    invalid_toml: str,
) -> None:
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(invalid_toml, encoding="utf-8")

    with pytest.raises(TypeError):
        _ = get_config()


def test_get_config_resume_blocks_forbidden_hyperparameters(
    tmp_path: pathlib.Path,
    valid_toml_content: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(valid_toml_content, encoding="utf-8")

    custom_args = [
        "test_program",
        "--resume",
        str(tmp_path),
        "--population-size",
        "100",
    ]
    monkeypatch.setattr(sys, "argv", custom_args)

    with pytest.raises(
        ValueError,
        match="Configuration Conflict: The following flags cannot be changed",
    ):
        _ = get_config()


def test_get_config_resume_allows_extending_generations_and_saves(
    tmp_path: pathlib.Path,
    valid_toml_content: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(valid_toml_content, encoding="utf-8")

    custom_args = [
        "test_program",
        "--resume",
        str(tmp_path),
        "--generations",
        "500",
        "--save-interval",
        "25",
    ]
    monkeypatch.setattr(sys, "argv", custom_args)

    result = get_config()
    assert result.resume == str(tmp_path)
    assert result.generations == 500
    assert result.save_interval == 25
