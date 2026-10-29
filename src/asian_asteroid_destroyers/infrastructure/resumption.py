import importlib
import pickle
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np

from asian_asteroid_destroyers.genetic_algorithm import (
    EvolutionAlgorithm,
    TrainingResult,
)
from asian_asteroid_destroyers.infrastructure.checkpointing import (
    CheckpointPayload,
)


class ResumptionWorkspace(NamedTuple):
    algorithm: EvolutionAlgorithm[TrainingResult]
    last_generation: int
    saved_state: TrainingResult
    rng: np.random.Generator


def _load_algorithm(path_str: str) -> EvolutionAlgorithm[TrainingResult]:
    try:
        module_path, function_name = path_str.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return cast(
            EvolutionAlgorithm[TrainingResult],
            getattr(module, function_name),
        )
    except (ValueError, AttributeError, ModuleNotFoundError) as error:
        raise ImportError(
            "Failed to dynamically import the target evolution module "
            + f"from checkpoint: {path_str}"
        ) from error


def load_resumption_workspace(
    experiment_directory: Path,
) -> ResumptionWorkspace:
    pkl_files = list(experiment_directory.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No valid state snapshot (.pkl) detected in {experiment_directory}"
        )

    emergency_file = experiment_directory / "emergency_runtime.pkl"
    target_checkpoint = (
        emergency_file if emergency_file.exists() else max(pkl_files)
    )

    print(f"Reading simulation workspace metadata: {target_checkpoint.name}")
    with open(target_checkpoint, "rb") as file:
        payload = cast(CheckpointPayload, pickle.load(file))

    algorithm = _load_algorithm(payload["algorithm_path"])
    last_generation = payload["generation"]
    saved_state = payload["state"]
    bit_generator = np.random.PCG64()
    setattr(bit_generator, "state", payload["rng_state"])
    rng = np.random.Generator(bit_generator)

    return ResumptionWorkspace(
        algorithm=algorithm,
        last_generation=last_generation,
        saved_state=saved_state,
        rng=rng,
    )
