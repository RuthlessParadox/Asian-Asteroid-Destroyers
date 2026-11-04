import argparse
import sys
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
    scenarios_path: str


@dataclass(frozen=True)
class TrainerArgs:
    generations: int
    population_size: int
    save_interval: int
    cx_pb: float
    mut_pb: float
    gene_size: int
    scenarios_path: Path
    resume: str | None = None


@dataclass(frozen=True)
class AnalyzerArgs:
    run_dir: str
    track_genes: list[int]


@dataclass(frozen=True)
class EvaluationArgs:
    history_path: Path
    logbook_path: Path


def get_trainer_config() -> TrainerArgs:
    return _parse_trainer_args(
        _load_baseline_config(Path("pyproject.toml")),
    )


def get_analyzer_config() -> AnalyzerArgs:
    parser = argparse.ArgumentParser(
        description="Extract metrics and render analytics for a training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to targeted experiment workspace directory",
    )
    _ = parser.add_argument(
        "--track-genes",
        type=int,
        nargs="+",
        default=[],
        help="Space-separated list of flat gene indices to trace for drift",
    )

    namespace = parser.parse_args()

    run_dir = cast(str, namespace.run_dir)
    track_genes = cast(list[int], namespace.track_genes)

    return AnalyzerArgs(
        run_dir=run_dir,
        track_genes=track_genes,
    )


def get_evaluation_config() -> EvaluationArgs:
    parser = argparse.ArgumentParser(
        description="Evaluate the performance of the elite individual from training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = parser.add_argument(
        "--history-path",
        type=str,
        required=True,
        help="Path to the final_history.parquet file",
    )
    _ = parser.add_argument(
        "--logbook-path",
        type=str,
        required=True,
        help="Path to the final_logbook.parquet file",
    )

    namespace = parser.parse_args()

    history_path = Path(cast(str, namespace.history_path))
    logbook_path = Path(cast(str, namespace.logbook_path))

    if not history_path.is_file():
        raise FileNotFoundError(f"History file not found at: {history_path}")
    if not logbook_path.is_file():
        raise FileNotFoundError(f"Logbook file not found at: {logbook_path}")

    return EvaluationArgs(
        history_path=history_path,
        logbook_path=logbook_path,
    )


def _parse_trainer_args(baseline: _EvolutionConfig) -> TrainerArgs:
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
    _ = parser.add_argument(
        "--scenarios-path",
        type=str,
        default=baseline.scenarios_path,
        help="Path to JSON containing training scenarios",
    )
    _ = parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an experiment directory to resume training from",
    )

    namespace = parser.parse_args()

    generations = cast(int, namespace.generations)
    population_size = cast(int, namespace.population_size)
    save_interval = cast(int, namespace.save_interval)
    cx_pb = cast(float, namespace.cx_pb)
    mut_pb = cast(float, namespace.mut_pb)
    gene_size = cast(int, namespace.gene_size)
    scenarios_path = cast(str, namespace.scenarios_path)
    resume = cast(str | None, namespace.resume)

    if resume is not None:
        forbidden_resume_flags = {
            "--population-size",
            "--cx-pb",
            "--mut-pb",
            "--gene-size",
        }
        conflicting_flags = forbidden_resume_flags.intersection(sys.argv)
        if conflicting_flags:
            flag_str = ", ".join(conflicting_flags)
            raise ValueError(
                "Configuration Conflict: The following flags cannot be "
                + f"changed when resuming an existing run: [{flag_str}]. "
                + "To alter hyperparameters, you must initiate "
                + "a fresh training run."
            )

    return TrainerArgs(
        generations=generations,
        population_size=population_size,
        save_interval=save_interval,
        cx_pb=cx_pb,
        mut_pb=mut_pb,
        gene_size=gene_size,
        scenarios_path=Path(scenarios_path),
        resume=resume,
    )


def _load_baseline_config(config_path: Path) -> _EvolutionConfig:
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

    scenarios_path = evolution_table["scenarios_path"]
    if not isinstance(scenarios_path, str):
        raise TypeError("scenarios_path in pyproject.toml must be a string")

    return _EvolutionConfig(
        generations=generations,
        population_size=population_size,
        save_interval=save_interval,
        cx_pb=cx_pb,
        mut_pb=mut_pb,
        gene_size=gene_size,
        scenarios_path=scenarios_path,
    )
