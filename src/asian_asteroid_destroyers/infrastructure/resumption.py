import gzip
import importlib
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import polars as pl

from asian_asteroid_destroyers.evokit import (
    EvolutionAlgorithm,
    HistoryLogger,
    Logbook,
    TrainingResult,
)

from .checkpointing import (
    EMERGENCY_HISTORY_FILENAME,
    EMERGENCY_LOGBOOK_FILENAME,
    EMERGENCY_RUNTIME_FILENAME,
    PERIODIC_RUNTIME_PATTERN,
    CheckpointPayload,
    get_periodic_history_filename,
    get_periodic_logbook_filename,
)
from .unpickling import MainHydratingUnpickler


class ResumptionWorkspace(NamedTuple):
    algorithm: EvolutionAlgorithm[TrainingResult]
    last_generation: int
    saved_state: TrainingResult
    rng: np.random.Generator


def _extract_generation_number(path: Path) -> int:
    try:
        raw_generation = path.stem.split("_gen_")[1].split(".")[0]
        return int(raw_generation)
    except (IndexError, ValueError) as error:
        raise ValueError(
            "Malformed or corrupted checkpoint filename encountered: "
            + f"'{path.name}'. Ensure file follows the "
            + "'runtime_gen_<integer>.pkl.gz' convention."
        ) from error


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
    pkl_files = list(experiment_directory.glob(PERIODIC_RUNTIME_PATTERN))

    emergency_file = experiment_directory / EMERGENCY_RUNTIME_FILENAME

    if not pkl_files and not emergency_file.exists():
        raise FileNotFoundError(
            f"No valid state snapshots detected in {experiment_directory}"
        )

    target_checkpoint = (
        emergency_file
        if emergency_file.exists()
        else max(pkl_files, key=_extract_generation_number)
    )

    print(f"Reading simulation workspace metadata: {target_checkpoint.name}")
    with gzip.open(target_checkpoint, "rb") as file:
        payload = cast(
            CheckpointPayload,
            MainHydratingUnpickler(file).load(),
        )

    algorithm = _load_algorithm(payload["algorithm_path"])

    last_generation = payload["generation"]

    if target_checkpoint.name == EMERGENCY_RUNTIME_FILENAME:
        logbook_path = experiment_directory / EMERGENCY_LOGBOOK_FILENAME
        history_path = experiment_directory / EMERGENCY_HISTORY_FILENAME
    else:
        logbook_path = experiment_directory / get_periodic_logbook_filename(
            last_generation,
        )
        history_path = experiment_directory / get_periodic_history_filename(
            last_generation,
        )
    rebuilt_logbook = None
    if logbook_path.exists():
        rebuilt_logbook = Logbook()
        rebuilt_logbook.hydrate(pl.read_parquet(logbook_path))
    rebuilt_history = None
    if history_path.exists():
        rebuilt_history = HistoryLogger()
        rebuilt_history.hydrate(pl.read_parquet(history_path))
    hydrated_state = TrainingResult(
        final_population=payload["final_population"],
        logbook=rebuilt_logbook,
        history_logger=rebuilt_history,
        hall_of_fame=payload["hall_of_fame"],
    )

    bit_generator = np.random.PCG64()
    setattr(bit_generator, "state", payload["rng_state"])
    rng = np.random.Generator(bit_generator)

    return ResumptionWorkspace(
        algorithm=algorithm,
        last_generation=last_generation,
        saved_state=hydrated_state,
        rng=rng,
    )
