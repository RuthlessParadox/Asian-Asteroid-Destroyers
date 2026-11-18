from pathlib import Path

import pytest

from asian_asteroid_destroyers.config import (
    EvolutionConfig,
    load_baseline_config,
)


def test_load_baseline_config_success(tmp_path: Path) -> None:
    toml_content = """
    [tool.evolution]
    generations = 100
    population_size = 50
    save_interval = 10
    cx_pb = 0.8
    mut_pb = 0.2
    gene_size = 10
    scenarios_path = "scenarios/default.json"
    """
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(toml_content)

    config = load_baseline_config(config_file)
    assert config == EvolutionConfig(
        generations=100,
        population_size=50,
        save_interval=10,
        cx_pb=0.8,
        mut_pb=0.2,
        gene_size=10,
        scenarios_path=Path("scenarios/default.json"),
    )


def test_load_baseline_config_file_not_found(tmp_path: Path) -> None:
    missing_file = tmp_path / "nonexistent.toml"
    with pytest.raises(
        FileNotFoundError,
        match="Required configuration file missing",
    ):
        _ = load_baseline_config(missing_file)


def test_load_baseline_config_missing_evolution_table(tmp_path: Path) -> None:
    toml_content = """
    [tool.other]
    some_key = "value"
    """
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(toml_content)

    with pytest.raises(
        KeyError,
        match="Missing required \\[tool\\.evolution\\] table",
    ):
        _ = load_baseline_config(config_file)


@pytest.mark.parametrize(
    "invalid_toml,expected_error_message",
    [
        (
            """
            [tool.evolution]
            generations = "not-an-int"
            population_size = 50
            save_interval = 10
            cx_pb = 0.8
            mut_pb = 0.2
            gene_size = 10
            scenarios_path = "path"
            """,
            "generations in pyproject.toml must be an integer",
        ),
        (
            """
            [tool.evolution]
            generations = 100
            population_size = "not-an-int"
            save_interval = 10
            cx_pb = 0.8
            mut_pb = 0.2
            gene_size = 10
            scenarios_path = "path"
            """,
            "population_size in pyproject.toml must be an integer",
        ),
        (
            """
            [tool.evolution]
            generations = 100
            population_size = 50
            save_interval = "not-an-int"
            cx_pb = 0.8
            mut_pb = 0.2
            gene_size = 10
            scenarios_path = "path"
            """,
            "save_interval in pyproject.toml must be an integer",
        ),
        (
            """
            [tool.evolution]
            generations = 100
            population_size = 50
            save_interval = 10
            cx_pb = "not-a-float"
            mut_pb = 0.2
            gene_size = 10
            scenarios_path = "path"
            """,
            "cx_pb in pyproject.toml must be a float",
        ),
        (
            """
            [tool.evolution]
            generations = 100
            population_size = 50
            save_interval = 10
            cx_pb = 0.8
            mut_pb = "not-a-float"
            gene_size = 10
            scenarios_path = "path"
            """,
            "mut_pb in pyproject.toml must be a float",
        ),
        (
            """
            [tool.evolution]
            generations = 100
            population_size = 50
            save_interval = 10
            cx_pb = 0.8
            mut_pb = 0.2
            gene_size = "not-an-int"
            scenarios_path = "path"
            """,
            "gene_size in pyproject.toml must be an integer",
        ),
        (
            """
            [tool.evolution]
            generations = 100
            population_size = 50
            save_interval = 10
            cx_pb = 0.8
            mut_pb = 0.2
            gene_size = 10
            scenarios_path = 12345
            """,
            "scenarios_path in pyproject.toml must be a string",
        ),
    ],
)
def test_load_baseline_config_type_errors(
    tmp_path: Path,
    invalid_toml: str,
    expected_error_message: str,
) -> None:
    config_file = tmp_path / "pyproject.toml"
    _ = config_file.write_text(invalid_toml)

    with pytest.raises(TypeError, match=expected_error_message):
        _ = load_baseline_config(config_file)
