from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from typing_extensions import override

from asian_asteroid_destroyers.evokit import (
    HallOfFame,
    HistoryLogger,
    Logbook,
    Statistics,
    TrainingResult,
    build_toolbox,
    get_individual_class,
)
from asian_asteroid_destroyers.infrastructure import (
    POINTER_FILENAME,
    get_periodic_history_filename,
    get_periodic_logbook_filename,
    get_periodic_runtime_filename,
    make_checkpoint_saver,
)

if TYPE_CHECKING:
    import pytest

    from asian_asteroid_destroyers.evokit import GenerationStats, Population
    from asian_asteroid_destroyers.infrastructure import CheckpointPayload


def _dummy_algorithm(
    toolbox: object,
    cx_pb: float,
    mut_pb: float,
    generations: int,
    population_size: int,
    *,
    seed: int | np.random.Generator | None = None,
    master_seed: int | None = None,
    workers: int | None = None,
    hall_of_fame: HallOfFame | None = None,
    statistics: Statistics | None = None,
    logbook: Logbook | None = None,
    history_logger: HistoryLogger | None = None,
    checkpoint_callback: object = None,
    initial_population: Population | None = None,
    start_generation: int = 1,
) -> TrainingResult:
    _ = (
        toolbox,
        cx_pb,
        mut_pb,
        generations,
        population_size,
        seed,
        master_seed,
        workers,
        hall_of_fame,
        statistics,
        logbook,
        history_logger,
        checkpoint_callback,
        initial_population,
        start_generation,
    )
    raise NotImplementedError("Dummy algorithm stub for type-checking")


def _make_population(
    weights: tuple[float, ...] = (1.0,),
    size: int = 3,
) -> tuple[Population, HallOfFame]:
    individual_class = get_individual_class(weights)
    toolbox = build_toolbox(
        evaluation_function=lambda _individual, _seed: (
            float(sum(_individual)),
        ),
        gene_size=2,
        weights=weights,
        individual_factory=lambda: individual_class([0.5, 0.25]),
    )
    population = list(toolbox.population(size))
    for index, individual in enumerate(population):
        individual.fitness.values = (float(index + 1),)
    hall_of_fame = HallOfFame()
    hall_of_fame.update(population)
    return population, hall_of_fame


def _make_training_result(
    population: Population,
    hall_of_fame: HallOfFame | None,
    *,
    with_logbook: bool = True,
    with_history: bool = True,
    generation: int = 1,
) -> TrainingResult:
    statistics = Statistics()
    logbook: Logbook | None = None
    if with_logbook:
        logbook = Logbook()
        assert logbook is not None
        generation_stats: GenerationStats = statistics.compile(population)
        logbook.record(generation, generation_stats)

    history_logger: HistoryLogger | None = None
    if with_history and hall_of_fame is not None:
        history_logger = HistoryLogger()
        assert history_logger is not None
        history_logger.record(generation, hall_of_fame)

    return TrainingResult(
        final_population=population,
        logbook=logbook,
        history_logger=history_logger,
        hall_of_fame=hall_of_fame,
    )


def _read_checkpoint(path: Path) -> CheckpointPayload:
    with gzip.open(path, "rb") as file:
        loaded: CheckpointPayload = (  # pyright: ignore[reportAny]
            pickle.load(file)
        )
    return loaded


def test_creates_nested_output_directory(tmp_path: Path) -> None:
    target_directory = tmp_path / "nested" / "checkpoints"
    rng = np.random.default_rng(42)

    _ = make_checkpoint_saver(
        output_directory=target_directory,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=1234,
        save_interval=5,
    )

    assert target_directory.is_dir()


def test_skips_non_interval_generations(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=1234,
        save_interval=10,
    )
    population, hall_of_fame = _make_population()
    state = _make_training_result(population, hall_of_fame, generation=3)

    callback(3, state)

    assert list(tmp_path.iterdir()) == []


def test_saves_runtime_logbook_history_and_pointer_on_interval(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(99)
    master_seed = 777
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=master_seed,
        save_interval=10,
    )
    population, hall_of_fame = _make_population()
    state = _make_training_result(population, hall_of_fame, generation=10)

    callback(10, state)

    runtime_name = get_periodic_runtime_filename(10)
    runtime_path = tmp_path / runtime_name
    logbook_path = tmp_path / get_periodic_logbook_filename(10)
    history_path = tmp_path / get_periodic_history_filename(10)
    pointer_path = tmp_path / POINTER_FILENAME

    assert runtime_path.is_file()
    assert logbook_path.is_file()
    assert history_path.is_file()
    assert pointer_path.is_file()
    assert pointer_path.read_text(encoding="utf-8") == runtime_name

    payload = _read_checkpoint(runtime_path)
    expected_algorithm_path = (
        f"{_dummy_algorithm.__module__}.{_dummy_algorithm.__name__}"
    )
    assert payload["algorithm_path"] == expected_algorithm_path
    assert payload["generation"] == 10
    assert payload["master_seed"] == master_seed
    payload_hall_of_fame = payload["hall_of_fame"]
    assert payload_hall_of_fame is not None
    assert len(payload_hall_of_fame) == len(hall_of_fame)
    assert len(payload["final_population"]) == len(population)
    assert (
        payload["rng_state"]["bit_generator"]
        == rng.bit_generator.state["bit_generator"]
    )

    logbook_df = pl.read_parquet(logbook_path)  # type: ignore[arg-type]
    assert logbook_df["generation"].to_list() == [10]

    history_df = pl.read_parquet(history_path)  # type: ignore[arg-type]
    assert history_df["generation"].to_list() == [10]


def test_handles_optional_none_logbook_and_history_logger(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(1)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=10,
        save_interval=5,
    )
    population, hall_of_fame = _make_population()
    state = _make_training_result(
        population,
        hall_of_fame,
        with_logbook=False,
        with_history=False,
        generation=5,
    )

    callback(5, state)

    assert (tmp_path / get_periodic_runtime_filename(5)).is_file()
    assert not (tmp_path / get_periodic_logbook_filename(5)).exists()
    assert not (tmp_path / get_periodic_history_filename(5)).exists()


def test_pruning_keeps_current_previous_and_milestones_only(
    tmp_path: Path,
) -> None:
    save_interval = 10
    rng = np.random.default_rng(2026)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=42,
        save_interval=save_interval,
    )
    population, hall_of_fame = _make_population()

    for generation in range(10, 121, 10):
        state = _make_training_result(
            population,
            hall_of_fame,
            generation=generation,
        )
        callback(generation, state)

    expected_kept_generations = [100, 110, 120]
    for generation in expected_kept_generations:
        assert (tmp_path / get_periodic_runtime_filename(generation)).is_file()
        assert (tmp_path / get_periodic_logbook_filename(generation)).is_file()
        assert (tmp_path / get_periodic_history_filename(generation)).is_file()

    for pruned_generation in range(10, 100, 10):
        assert not (
            tmp_path / get_periodic_runtime_filename(pruned_generation)
        ).exists()
        assert not (
            tmp_path / get_periodic_logbook_filename(pruned_generation)
        ).exists()
        assert not (
            tmp_path / get_periodic_history_filename(pruned_generation)
        ).exists()

    assert (tmp_path / POINTER_FILENAME).read_text(
        encoding="utf-8"
    ) == get_periodic_runtime_filename(120)


def test_ignores_malformed_runtime_filename_during_pruning(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=7,
        save_interval=10,
    )
    malformed = tmp_path / "runtime_gen_corrupted.pkl.gz"
    _ = malformed.write_bytes(b"not-a-valid-checkpoint")

    population, hall_of_fame = _make_population()
    for generation in (10, 20, 30):
        callback(
            generation,
            _make_training_result(
                population,
                hall_of_fame,
                generation=generation,
            ),
        )

    assert malformed.is_file()


def test_pruning_continues_if_unlink_raises_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(8)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=8,
        save_interval=10,
    )
    population, hall_of_fame = _make_population()
    for generation in (10, 20):
        callback(
            generation,
            _make_training_result(
                population,
                hall_of_fame,
                generation=generation,
            ),
        )

    stale_runtime = tmp_path / get_periodic_runtime_filename(10)
    assert stale_runtime.is_file()

    original_unlink = Path.unlink

    def _failing_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == stale_runtime:
            raise OSError("Simulated permission error")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _failing_unlink)

    callback(
        30,
        _make_training_result(population, hall_of_fame, generation=30),
    )

    assert stale_runtime.is_file()
    assert (tmp_path / get_periodic_runtime_filename(30)).is_file()


def test_pointer_write_failure_does_not_abort_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(12)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=12,
        save_interval=5,
    )
    population, hall_of_fame = _make_population()
    state = _make_training_result(population, hall_of_fame, generation=5)

    def _failing_replace(self: Path, target: Path) -> Path:
        _ = (self, target)
        raise OSError("Simulated atomic pointer rename failure")

    monkeypatch.setattr(Path, "replace", _failing_replace)

    callback(5, state)

    runtime_path = tmp_path / get_periodic_runtime_filename(5)
    assert runtime_path.is_file()
    assert not (tmp_path / POINTER_FILENAME).exists()


def test_custom_save_interval_and_milestone_retention(tmp_path: Path) -> None:
    save_interval = 4
    rng = np.random.default_rng(55)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=999,
        save_interval=save_interval,
    )
    population, hall_of_fame = _make_population()

    for generation in range(4, 49, 4):
        callback(
            generation,
            _make_training_result(
                population,
                hall_of_fame,
                generation=generation,
            ),
        )

    for kept_generation in (40, 44, 48):
        assert (
            tmp_path / get_periodic_runtime_filename(kept_generation)
        ).is_file()

    for pruned_generation in range(4, 37, 4):
        assert not (
            tmp_path / get_periodic_runtime_filename(pruned_generation)
        ).exists()


def test_saved_rng_state_can_restore_generator(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=123,
        save_interval=1,
    )
    population, hall_of_fame = _make_population()

    _ = rng.random()
    callback(1, _make_training_result(population, hall_of_fame, generation=1))

    expected_next_draws = [float(rng.random()) for _ in range(3)]

    payload = _read_checkpoint(tmp_path / get_periodic_runtime_filename(1))
    restored_rng = np.random.Generator(np.random.PCG64())
    restored_rng.bit_generator.state = payload["rng_state"]

    restored_draws = [float(restored_rng.random()) for _ in range(3)]
    assert restored_draws == expected_next_draws


def test_generation_zero_is_checkpointed_and_treated_as_milestone(
    tmp_path: Path,
) -> None:
    save_interval = 10
    rng = np.random.default_rng(303)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=303,
        save_interval=save_interval,
    )
    population, hall_of_fame = _make_population()

    callback(0, _make_training_result(population, hall_of_fame, generation=0))
    for generation in (10, 20, 30):
        callback(
            generation,
            _make_training_result(
                population,
                hall_of_fame,
                generation=generation,
            ),
        )

    assert (tmp_path / get_periodic_runtime_filename(0)).is_file()
    assert not (tmp_path / get_periodic_runtime_filename(10)).exists()
    assert (tmp_path / get_periodic_runtime_filename(20)).is_file()
    assert (tmp_path / get_periodic_runtime_filename(30)).is_file()


def test_pruning_succeeds_without_logbook_or_history_files(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(404)
    callback = make_checkpoint_saver(
        output_directory=tmp_path,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=404,
        save_interval=10,
    )
    population, hall_of_fame = _make_population()

    for generation in (10, 20, 30):
        state = _make_training_result(
            population,
            hall_of_fame,
            with_logbook=False,
            with_history=False,
            generation=generation,
        )
        callback(generation, state)

    assert not (tmp_path / get_periodic_runtime_filename(10)).exists()
    assert (tmp_path / get_periodic_runtime_filename(20)).is_file()
    assert (tmp_path / get_periodic_runtime_filename(30)).is_file()


def test_accepts_pathlike_output_directory(tmp_path: Path) -> None:
    class _CustomPathLike(os.PathLike[str]):
        _target: Path

        def __init__(self, target: Path) -> None:
            self._target = target

        @override
        def __fspath__(self) -> str:
            return os.fspath(self._target)

    pathlike_directory = _CustomPathLike(tmp_path / "pathlike_out")
    rng = np.random.default_rng(505)
    callback = make_checkpoint_saver(
        output_directory=pathlike_directory,
        rng=rng,
        algorithm=_dummy_algorithm,
        master_seed=505,
        save_interval=2,
    )
    population, hall_of_fame = _make_population()

    callback(2, _make_training_result(population, hall_of_fame, generation=2))

    assert (
        tmp_path / "pathlike_out" / get_periodic_runtime_filename(2)
    ).is_file()
