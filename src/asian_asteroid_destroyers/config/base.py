from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import tomli

if TYPE_CHECKING:
    import os


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int
    population_size: int
    save_interval: int
    cx_pb: float
    mut_pb: float
    gene_size: int
    scenarios_path: Path


def load_baseline_config(
    config_path: str | os.PathLike[str] = "pyproject.toml",
) -> EvolutionConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Required configuration file missing: {config_path}"
        )

    with open(config_path, "rb") as file:
        toml_data = tomli.load(file)

    tool_table: dict[str, object] = (  # pyright: ignore[reportAny]
        toml_data.get("tool", {})
    )
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

    scenarios_path = evolution_table["scenarios_path"]
    if not isinstance(scenarios_path, str):
        raise TypeError("scenarios_path in pyproject.toml must be a string")

    return EvolutionConfig(
        generations=generations,
        population_size=population_size,
        save_interval=save_interval,
        cx_pb=cx_pb,
        mut_pb=mut_pb,
        gene_size=gene_size,
        scenarios_path=Path(scenarios_path),
    )
