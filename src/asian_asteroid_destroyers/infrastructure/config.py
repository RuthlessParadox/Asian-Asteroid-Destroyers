import argparse
import datetime
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import tomli


@dataclass(frozen=True)
class _EvolutionConfig:
    generations: int
    population_size: int
    save_interval: int
    cx_pb: float
    mut_pb: float
    gene_size: int


@dataclass(frozen=True)
class RuntimeArgs:
    generations: int
    population_size: int
    save_interval: int
    cx_pb: float
    mut_pb: float
    gene_size: int


def get_config() -> RuntimeArgs:
    return _parse_args(_load_baseline_config(pathlib.Path("pyproject.toml")))


def _parse_args(baseline: _EvolutionConfig) -> RuntimeArgs:
    parser = argparse.ArgumentParser(
        description="Run evolutionary training with layered hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = parser.add_argument(
        "--generations",
        type=int,
        default=baseline.generations,
        help="Number of evolutionary generations to simulate",
    )
    _ = parser.add_argument(
        "--population-size",
        type=int,
        default=baseline.population_size,
        help="Size of the individual population",
    )
    _ = parser.add_argument(
        "--save-interval",
        type=int,
        default=baseline.save_interval,
        help="Generation frequency for periodic checkpoints",
    )
    _ = parser.add_argument(
        "--cx-pb",
        type=float,
        default=baseline.cx_pb,
        help="Crossover probability coefficient",
    )
    _ = parser.add_argument(
        "--mut-pb",
        type=float,
        default=baseline.mut_pb,
        help="Mutation probability coefficient",
    )
    _ = parser.add_argument(
        "--gene-size",
        type=int,
        default=baseline.gene_size,
        help="Size of individual genetic configurations",
    )

    namespace = parser.parse_args()

    generations = cast(int, namespace.generations)
    population_size = cast(int, namespace.population_size)
    save_interval = cast(int, namespace.save_interval)
    cx_pb = cast(float, namespace.cx_pb)
    mut_pb = cast(float, namespace.mut_pb)
    gene_size = cast(int, namespace.gene_size)

    return RuntimeArgs(
        generations=generations,
        population_size=population_size,
        save_interval=save_interval,
        cx_pb=cx_pb,
        mut_pb=mut_pb,
        gene_size=gene_size,
    )


def _load_baseline_config(config_path: pathlib.Path) -> _EvolutionConfig:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Required configuration file missing: {config_path}"
        )

    with open(config_path, "rb") as f:
        toml_data = tomli.load(f)

    tool_table = cast(dict[str, object], toml_data.get("tool", {}))
    evolution_table = cast(dict[str, object], tool_table.get("evolution", {}))
    if not evolution_table:
        raise KeyError(
            "Missing required [tool.evolution] table in pyproject.toml"
        )

    generations = evolution_table["generations"]
    if not isinstance(generations, int):
        raise TypeError("generations in pyproject.toml must be an integer")

    population_size = evolution_table["population_size"]
    if not isinstance(population_size, int):
        raise TypeError("population_size in pyproject.toml must be an integer")

    save_interval = evolution_table["save_interval"]
    if not isinstance(save_interval, int):
        raise TypeError("save_interval in pyproject.toml must be an integer")

    cx_pb = evolution_table["cx_pb"]
    if not isinstance(cx_pb, float):
        raise TypeError("cx_pb in pyproject.toml must be a float")

    mut_pb = evolution_table["mut_pb"]
    if not isinstance(mut_pb, float):
        raise TypeError("mut_pb in pyproject.toml must be a float")

    gene_size = evolution_table["gene_size"]
    if not isinstance(gene_size, int):
        raise TypeError("gene_size in pyproject.toml must be an integer")

    return _EvolutionConfig(
        generations=generations,
        population_size=population_size,
        save_interval=save_interval,
        cx_pb=cx_pb,
        mut_pb=mut_pb,
        gene_size=gene_size,
    )


def create_experiment_directory(base_directory: str | Path = "./data") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = Path(base_directory) / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory
