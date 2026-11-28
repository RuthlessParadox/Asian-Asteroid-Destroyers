from __future__ import annotations

import gzip
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from asian_asteroid_destroyers.evokit import (
    HallOfFame,
    HistoryLogger,
    Logbook,
    Statistics,
    build_toolbox,
    get_individual_class,
    run_nsga_evolution,
)
from asian_asteroid_destroyers.infrastructure import (
    EMERGENCY_HISTORY_FILENAME,
    EMERGENCY_LOGBOOK_FILENAME,
    EMERGENCY_RUNTIME_FILENAME,
    ResumptionWorkspace,
    get_periodic_history_filename,
    get_periodic_logbook_filename,
    get_periodic_runtime_filename,
    load_resumption_workspace,
    make_checkpoint_saver,
    update_active_pointer,
)

if TYPE_CHECKING:
    from pathlib import Path

    from asian_asteroid_destroyers.evokit import (
        Individual,
        Population,
        TrainingResult,
    )
    from asian_asteroid_destroyers.infrastructure import CheckpointPayload


def _algorithm_path(function: object) -> str:
    module_name = getattr(function, "__module__", "")
    qual_name = getattr(function, "__name__", "")
    return f"{module_name}.{qual_name}"


def _make_population(
    weights: tuple[float, ...] = (1.0, -1.0),
    count: int = 3,
) -> Population:
    individual_class = get_individual_class(weights)
    population: list[Individual] = []
    for index in range(count):
        individual = individual_class(
            [float(index) * 0.25, 1.0 - float(index) * 0.25]
        )
        individual.fitness.values = (float(index + 1), -float(index))
        population.append(individual)
    return population


def _write_checkpoint(
    path: Path,
    *,
    algorithm_path: str,
    generation: int,
    final_population: Population,
    hall_of_fame: HallOfFame | None,
    rng: np.random.Generator,
    master_seed: int,
) -> None:
    rng_state = rng.bit_generator.state
    assert isinstance(rng_state, dict)
    payload: CheckpointPayload = {
        "algorithm_path": algorithm_path,
        "generation": generation,
        "final_population": final_population,
        "hall_of_fame": hall_of_fame,
        "rng_state": rng_state,
        "master_seed": master_seed,
    }
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _build_logbook_and_history(
    population: Population,
    generations: list[int],
) -> tuple[Logbook, HistoryLogger]:
    statistics = Statistics()
    hall_of_fame = HallOfFame()
    hall_of_fame.update(population)

    logbook = Logbook()
    history = HistoryLogger()
    for generation in generations:
        logbook.record(generation, statistics.compile(population))
        history.record(generation, hall_of_fame)
    return logbook, history


def test_load_resumption_workspace_from_active_pointer(tmp_path: Path) -> None:
    population = _make_population(weights=(1.0, -1.0), count=3)
    hall_of_fame = HallOfFame()
    hall_of_fame.update(population)
    logbook, history = _build_logbook_and_history(population, [5, 10])

    rng = np.random.default_rng(12345)
    _ = rng.random(4)

    generation_10_name = get_periodic_runtime_filename(10)
    generation_20_name = get_periodic_runtime_filename(20)
    _write_checkpoint(
        tmp_path / generation_10_name,
        algorithm_path=_algorithm_path(run_nsga_evolution),
        generation=10,
        final_population=population,
        hall_of_fame=hall_of_fame,
        rng=rng,
        master_seed=999,
    )
    _write_checkpoint(
        tmp_path / generation_20_name,
        algorithm_path=_algorithm_path(run_nsga_evolution),
        generation=20,
        final_population=population,
        hall_of_fame=hall_of_fame,
        rng=rng,
        master_seed=999,
    )
    logbook.save_parquet(tmp_path / get_periodic_logbook_filename(10))
    history.save_parquet(tmp_path / get_periodic_history_filename(10))

    update_active_pointer(tmp_path, generation_10_name)

    workspace = load_resumption_workspace(tmp_path)

    assert isinstance(workspace, ResumptionWorkspace)
    assert workspace.algorithm is run_nsga_evolution
    assert workspace.last_generation == 10
    assert workspace.master_seed == 999
    assert len(workspace.saved_state.final_population) == 3
    assert (
        workspace.saved_state.final_population[0].fitness.values
        == population[0].fitness.values
    )
    saved_hall_of_fame = workspace.saved_state.hall_of_fame
    assert saved_hall_of_fame is not None
    assert len(saved_hall_of_fame) == len(hall_of_fame)

    saved_logbook = workspace.saved_state.logbook
    assert saved_logbook is not None
    assert saved_logbook.dataframe["generation"].to_list() == [5, 10]

    saved_history = workspace.saved_state.history_logger
    assert saved_history is not None
    assert saved_history.dataframe["generation"].to_list() == [5, 10]

    expected_next = rng.random(5)
    actual_next = workspace.rng.random(5)
    np.testing.assert_allclose(actual_next, expected_next)


def test_load_resumption_workspace_without_pointer_prefers_emergency(
    tmp_path: Path,
) -> None:
    population = _make_population(count=2)
    hall_of_fame = HallOfFame()
    hall_of_fame.update(population)
    logbook, history = _build_logbook_and_history(population, [7])
    rng = np.random.default_rng(77)

    _write_checkpoint(
        tmp_path / get_periodic_runtime_filename(5),
        algorithm_path=_algorithm_path(run_nsga_evolution),
        generation=5,
        final_population=population,
        hall_of_fame=None,
        rng=rng,
        master_seed=11,
    )
    _write_checkpoint(
        tmp_path / get_periodic_runtime_filename(12),
        algorithm_path=_algorithm_path(run_nsga_evolution),
        generation=12,
        final_population=population,
        hall_of_fame=None,
        rng=rng,
        master_seed=11,
    )
    _write_checkpoint(
        tmp_path / EMERGENCY_RUNTIME_FILENAME,
        algorithm_path=_algorithm_path(run_nsga_evolution),
        generation=7,
        final_population=population,
        hall_of_fame=hall_of_fame,
        rng=rng,
        master_seed=22,
    )
    logbook.save_parquet(tmp_path / EMERGENCY_LOGBOOK_FILENAME)
    history.save_parquet(tmp_path / EMERGENCY_HISTORY_FILENAME)

    workspace = load_resumption_workspace(tmp_path)

    assert workspace.last_generation == 7
    assert workspace.master_seed == 22
    saved_logbook = workspace.saved_state.logbook
    assert saved_logbook is not None
    assert saved_logbook.dataframe["generation"].to_list() == [7]
    saved_history = workspace.saved_state.history_logger
    assert saved_history is not None
    assert saved_history.dataframe["generation"].to_list() == [7]


def test_load_resumption_workspace_without_pointer_selects_highest_generation(
    tmp_path: Path,
) -> None:
    population = _make_population(count=2)
    rng = np.random.default_rng(55)

    for generation in (2, 15, 8):
        _write_checkpoint(
            tmp_path / get_periodic_runtime_filename(generation),
            algorithm_path=_algorithm_path(run_nsga_evolution),
            generation=generation,
            final_population=population,
            hall_of_fame=None,
            rng=rng,
            master_seed=300 + generation,
        )

    workspace = load_resumption_workspace(tmp_path)

    assert workspace.last_generation == 15
    assert workspace.master_seed == 315
    assert workspace.saved_state.logbook is None
    assert workspace.saved_state.history_logger is None
    assert workspace.saved_state.hall_of_fame is None


def test_load_resumption_workspace_missing_checkpoints_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _ = load_resumption_workspace(tmp_path)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "not_a_valid_module_path",
        "asian_asteroid_destroyers.evokit.nonexistent_module"
        + ".run_nsga_evolution",
        "asian_asteroid_destroyers.evokit.resumption.does_not_exist",
    ],
)
def test_load_resumption_workspace_invalid_algorithm_path_raises(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    population = _make_population(count=1)
    rng = np.random.default_rng(1)
    checkpoint_name = get_periodic_runtime_filename(1)
    _write_checkpoint(
        tmp_path / checkpoint_name,
        algorithm_path=invalid_path,
        generation=1,
        final_population=population,
        hall_of_fame=None,
        rng=rng,
        master_seed=1,
    )
    update_active_pointer(tmp_path, checkpoint_name)

    with pytest.raises(ImportError):
        _ = load_resumption_workspace(tmp_path)


def _simple_evaluator(individual: Individual, _seed: int) -> tuple[float,]:
    return (sum(individual),)


def test_checkpoint_saver_and_resumption_round_trip(tmp_path: Path) -> None:
    toolbox = build_toolbox(
        evaluation_function=_simple_evaluator,
        gene_size=3,
        weights=(1.0,),
    )
    rng = np.random.default_rng(42)
    saver = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=run_nsga_evolution,
        master_seed=4242,
        save_interval=2,
    )

    _ = run_nsga_evolution(
        toolbox=toolbox,
        cx_pb=0.5,
        mut_pb=0.2,
        generations=2,
        population_size=6,
        seed=rng,
        master_seed=4242,
        workers=1,
        hall_of_fame=HallOfFame(),
        statistics=Statistics(),
        logbook=Logbook(),
        history_logger=HistoryLogger(),
        checkpoint_callback=saver,
    )

    workspace = load_resumption_workspace(tmp_path)
    assert workspace.last_generation == 2
    assert workspace.master_seed == 4242

    resumed_result: TrainingResult = workspace.algorithm(
        toolbox=toolbox,
        cx_pb=0.5,
        mut_pb=0.2,
        generations=4,
        population_size=6,
        seed=workspace.rng,
        master_seed=workspace.master_seed,
        workers=1,
        hall_of_fame=workspace.saved_state.hall_of_fame,
        statistics=Statistics(),
        logbook=workspace.saved_state.logbook,
        history_logger=workspace.saved_state.history_logger,
        initial_population=workspace.saved_state.final_population,
        start_generation=workspace.last_generation + 1,
    )

    assert len(resumed_result.final_population) == 6
    resumed_logbook = resumed_result.logbook
    assert resumed_logbook is not None
    assert resumed_logbook.dataframe["generation"].to_list() == [0, 1, 2, 3, 4]
    resumed_history = resumed_result.history_logger
    assert resumed_history is not None
    assert resumed_history.dataframe["generation"].to_list() == [0, 1, 2, 3, 4]


def test_load_resumption_workspace_partial_parquet_artifacts(
    tmp_path: Path,
) -> None:
    population = _make_population(count=2)
    logbook, _ = _build_logbook_and_history(population, [4])
    rng = np.random.default_rng(101)

    checkpoint_name = get_periodic_runtime_filename(4)
    _write_checkpoint(
        tmp_path / checkpoint_name,
        algorithm_path=_algorithm_path(run_nsga_evolution),
        generation=4,
        final_population=population,
        hall_of_fame=None,
        rng=rng,
        master_seed=404,
    )
    logbook.save_parquet(tmp_path / get_periodic_logbook_filename(4))
    update_active_pointer(tmp_path, checkpoint_name)

    workspace = load_resumption_workspace(tmp_path)

    saved_logbook = workspace.saved_state.logbook
    assert saved_logbook is not None
    assert saved_logbook.dataframe["generation"].to_list() == [4]
    assert workspace.saved_state.history_logger is None


def test_load_resumption_workspace_negative_and_decimal_weights(
    tmp_path: Path,
) -> None:
    weights = (1.5, -0.25, 2.0)
    individual_class = get_individual_class(weights)
    individual = individual_class([0.1, 0.2, 0.3])
    individual.fitness.values = (10.5, -3.25, 8.0)

    rng = np.random.default_rng(202)
    checkpoint_name = get_periodic_runtime_filename(3)
    _write_checkpoint(
        tmp_path / checkpoint_name,
        algorithm_path=_algorithm_path(run_nsga_evolution),
        generation=3,
        final_population=[individual],
        hall_of_fame=None,
        rng=rng,
        master_seed=303,
    )
    update_active_pointer(tmp_path, checkpoint_name)

    workspace = load_resumption_workspace(tmp_path)

    restored_individual = workspace.saved_state.final_population[0]
    assert restored_individual.fitness.weights == weights
    assert restored_individual.fitness.values == (10.5, -3.25, 8.0)
    assert list(restored_individual) == pytest.approx([0.1, 0.2, 0.3])


def test_load_resumption_workspace_missing_pointed_checkpoint_raises(
    tmp_path: Path,
) -> None:
    update_active_pointer(tmp_path, get_periodic_runtime_filename(99))

    with pytest.raises(FileNotFoundError):
        _ = load_resumption_workspace(tmp_path)


def test_load_resumption_workspace_corrupt_checkpoint_raises(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / get_periodic_runtime_filename(1)
    _ = checkpoint_path.write_bytes(b"not-valid-gzip-or-pickle-bytes")
    update_active_pointer(tmp_path, checkpoint_path.name)

    with pytest.raises((OSError, EOFError, pickle.UnpicklingError)):
        _ = load_resumption_workspace(tmp_path)


def test_load_resumption_workspace_malformed_periodic_filename_raises(
    tmp_path: Path,
) -> None:
    bad_file = tmp_path / "runtime_gen_abc.pkl.gz"
    with gzip.open(bad_file, "wb") as handle:
        pickle.dump({}, handle)

    with pytest.raises(ValueError):
        _ = load_resumption_workspace(tmp_path)
