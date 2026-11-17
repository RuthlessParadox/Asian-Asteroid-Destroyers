from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .base import load_baseline_config
from .parsers import parse_non_negative_float, parse_positive_int


@dataclass(frozen=True)
class TrainerArgs:
    generations: int
    population_size: int
    save_interval: int
    cx_pb: float
    mut_pb: float
    gene_size: int
    scenarios_path: Path
    resume: Path | None = None


def get_trainer_config() -> TrainerArgs:
    baseline = load_baseline_config("pyproject.toml")
    parser = argparse.ArgumentParser(
        description="Run evolutionary training with layered hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = parser.add_argument(
        "--generations",
        type=parse_positive_int,
        default=baseline.generations,
        help="Number of evolutionary generations to simulate",
    )
    _ = parser.add_argument(
        "--population-size",
        type=parse_positive_int,
        default=baseline.population_size,
        help="Size of the individual population",
    )
    _ = parser.add_argument(
        "--save-interval",
        type=parse_positive_int,
        default=baseline.save_interval,
        help="Generation frequency for periodic checkpoints",
    )
    _ = parser.add_argument(
        "--cx-pb",
        type=parse_non_negative_float,
        default=baseline.cx_pb,
        help="Crossover probability coefficient",
    )
    _ = parser.add_argument(
        "--mut-pb",
        type=parse_non_negative_float,
        default=baseline.mut_pb,
        help="Mutation probability coefficient",
    )
    _ = parser.add_argument(
        "--gene-size",
        type=parse_positive_int,
        default=baseline.gene_size,
        help="Size of individual genetic configurations",
    )
    _ = parser.add_argument(
        "--scenarios-path",
        type=str,
        default=str(baseline.scenarios_path),
        help="Path to JSON containing training scenarios",
    )
    _ = parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an experiment directory to resume training from",
    )

    namespace = parser.parse_args()

    generations: int = namespace.generations  # pyright: ignore[reportAny]
    population_size: int = (  # pyright: ignore[reportAny]
        namespace.population_size
    )
    save_interval: int = namespace.save_interval  # pyright: ignore[reportAny]
    cx_pb: float = namespace.cx_pb  # pyright: ignore[reportAny]
    mut_pb: float = namespace.mut_pb  # pyright: ignore[reportAny]
    gene_size: int = namespace.gene_size  # pyright: ignore[reportAny]
    scenarios_path: str = (  # pyright: ignore[reportAny]
        namespace.scenarios_path
    )
    resume_str: str | None = namespace.resume  # pyright: ignore[reportAny]
    resume = Path(resume_str) if resume_str is not None else None

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
                + "To alter hyperparameters, you must initiate a fresh "
                + "training run."
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
