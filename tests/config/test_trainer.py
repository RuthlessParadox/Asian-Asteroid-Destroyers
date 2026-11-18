from pathlib import Path

import pytest

from asian_asteroid_destroyers.config import (
    EvolutionConfig,
    get_trainer_config,
)


@pytest.fixture
def mock_baseline() -> EvolutionConfig:
    return EvolutionConfig(
        generations=10,
        population_size=50,
        save_interval=5,
        cx_pb=0.5,
        mut_pb=0.1,
        gene_size=20,
        scenarios_path=Path("scenarios.json"),
    )


def test_get_trainer_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
    mock_baseline: EvolutionConfig,
) -> None:
    monkeypatch.setattr("sys.argv", ["trainer.py"])

    args = get_trainer_config(baseline=mock_baseline)
    assert args.generations == 10
    assert args.population_size == 50
    assert args.save_interval == 5
    assert args.cx_pb == 0.5
    assert args.mut_pb == 0.1
    assert args.gene_size == 20
    assert args.scenarios_path == Path("scenarios.json")
    assert args.resume is None


def test_get_trainer_config_custom_args(
    monkeypatch: pytest.MonkeyPatch,
    mock_baseline: EvolutionConfig,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "trainer.py",
            "--generations",
            "20",
            "--population-size",
            "100",
            "--save-interval",
            "10",
            "--cx-pb",
            "0.7",
            "--mut-pb",
            "0.2",
            "--gene-size",
            "30",
            "--scenarios-path",
            "custom_scenarios.json",
        ],
    )

    args = get_trainer_config(baseline=mock_baseline)
    assert args.generations == 20
    assert args.population_size == 100
    assert args.save_interval == 10
    assert args.cx_pb == 0.7
    assert args.mut_pb == 0.2
    assert args.gene_size == 30
    assert args.scenarios_path == Path("custom_scenarios.json")
    assert args.resume is None


def test_get_trainer_config_valid_resume_modifications(
    monkeypatch: pytest.MonkeyPatch,
    mock_baseline: EvolutionConfig,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "trainer.py",
            "--resume",
            "experiments/run_01",
            "--generations",
            "25",
            "--save-interval",
            "20",
            "--scenarios-path",
            "new_scenarios.json",
        ],
    )

    args = get_trainer_config(baseline=mock_baseline)
    assert args.resume == Path("experiments/run_01")
    assert args.generations == 25
    assert args.save_interval == 20
    assert args.scenarios_path == Path("new_scenarios.json")
    assert args.population_size == mock_baseline.population_size


@pytest.mark.parametrize(
    "conflicting_arg",
    ["--population-size", "--cx-pb", "--mut-pb", "--gene-size"],
)
def test_resume_conflict_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
    mock_baseline: EvolutionConfig,
    conflicting_arg: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "trainer.py",
            "--resume",
            "experiments/run_01",
            conflicting_arg,
            "100",
        ],
    )

    with pytest.raises(ValueError, match="Configuration Conflict"):
        _ = get_trainer_config(baseline=mock_baseline)


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--generations", "abc"],
        ["--generations", "-5"],
        ["--population-size", "not-an-int"],
        ["--save-interval", "1.5"],
        ["--cx-pb", "not-a-float"],
        ["--cx-pb", "-0.1"],
        ["--mut-pb", "invalid"],
        ["--gene-size", "-10"],
    ],
)
def test_invalid_argument_parsing_raises_system_exit(
    monkeypatch: pytest.MonkeyPatch,
    mock_baseline: EvolutionConfig,
    invalid_args: list[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["trainer.py"] + invalid_args)

    with pytest.raises(SystemExit) as exc_info:
        _ = get_trainer_config(baseline=mock_baseline)

    assert exc_info.value.code == 2
