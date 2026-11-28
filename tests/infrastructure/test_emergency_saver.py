from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from asian_asteroid_destroyers.evokit import (
    HallOfFame,
    HistoryLogger,
    Logbook,
    TrainingResult,
    get_individual_class,
)
from asian_asteroid_destroyers.infrastructure import (
    EMERGENCY_HISTORY_FILENAME,
    EMERGENCY_LOGBOOK_FILENAME,
    EMERGENCY_RUNTIME_FILENAME,
    POINTER_FILENAME,
    EmergencySaver,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from asian_asteroid_destroyers.evokit import EvolutionAlgorithm
    from asian_asteroid_destroyers.evokit.statistics import Statistics
    from asian_asteroid_destroyers.evokit.toolbox import Toolbox
    from asian_asteroid_destroyers.evokit.types import (
        GenerationStats,
        Population,
    )


def _make_sample_algorithm() -> EvolutionAlgorithm[TrainingResult]:
    def _algorithm(
        toolbox: Toolbox,
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
        checkpoint_callback: Callable[[int, TrainingResult], None]
        | None = None,
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
        raise NotImplementedError(
            "Stub algorithm for checkpoint metadata only."
        )

    return _algorithm


def _make_sample_population(
    weights: tuple[float, ...] = (1.0, -1.0),
) -> Population:
    individual_class = get_individual_class(weights)
    first = individual_class([0.25, 0.75])
    first.fitness.values = (2.5, -1.0)
    second = individual_class([0.5, 0.1])
    second.fitness.values = (1.2, -0.4)
    return [first, second]


def _make_stats(max_value: float) -> GenerationStats:
    return {
        "max_fitness": max_value,
        "min_fitness": 0.1,
        "mean_fitness": max_value / 2.0,
        "median_fitness": max_value / 2.0,
        "std_fitness": 0.25,
    }


def _read_gzip_pickle(path: Path) -> object:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)  # pyright: ignore[reportAny]


def test_output_directory_accepts_str_and_path(tmp_path: Path) -> None:
    str_directory = str(tmp_path / "str_out")
    path_directory = tmp_path / "path_out"
    rng = np.random.default_rng(123)
    algorithm = _make_sample_algorithm()

    saver_from_str = EmergencySaver(
        str_directory,
        rng,
        algorithm,
        master_seed=42,
    )
    saver_from_path = EmergencySaver(
        path_directory,
        rng,
        algorithm,
        master_seed=42,
    )

    assert isinstance(saver_from_str.output_directory, Path)
    assert saver_from_str.output_directory == Path(str_directory)
    assert saver_from_path.output_directory == path_directory


def test_handle_interrupt_without_state_exits_with_code_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_directory = tmp_path / "no_state_run"
    rng = np.random.default_rng(7)
    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=99,
    )

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(2, None)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    out: str = captured.out
    assert "Signal 2" in out
    assert "No generation checkpoints available yet" in out
    assert not output_directory.exists()


def test_handle_interrupt_saves_runtime_population_hof_and_rng_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_directory = tmp_path / "nested" / "checkpoint_dir"
    rng = np.random.default_rng(2026)
    _ = rng.random(5)
    expected_rng_state = rng.bit_generator.state
    expected_next_draw = float(rng.random())

    rng = np.random.default_rng(2026)
    _ = rng.random(5)

    population = _make_sample_population((1.0, -1.0))
    hall_of_fame = HallOfFame()
    hall_of_fame.update(population)

    state = TrainingResult(
        final_population=population,
        logbook=None,
        history_logger=None,
        hall_of_fame=hall_of_fame,
    )

    algorithm = _make_sample_algorithm()
    saver = EmergencySaver(output_directory, rng, algorithm, master_seed=777)
    saver.update_state(14, state)

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(15, None)

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    out: str = captured.out
    assert "Securing data from generation 14" in out
    assert "Emergency checkpoint sequence completed" in out

    runtime_path = output_directory / EMERGENCY_RUNTIME_FILENAME
    pointer_path = output_directory / POINTER_FILENAME
    assert runtime_path.is_file()
    assert pointer_path.is_file()
    assert (
        pointer_path.read_text(encoding="utf-8") == EMERGENCY_RUNTIME_FILENAME
    )
    assert not (output_directory / EMERGENCY_LOGBOOK_FILENAME).exists()
    assert not (output_directory / EMERGENCY_HISTORY_FILENAME).exists()

    loaded = _read_gzip_pickle(runtime_path)
    assert isinstance(loaded, dict)
    assert (
        loaded["algorithm_path"]
        == f"{algorithm.__module__}.{algorithm.__name__}"
    )
    assert loaded["generation"] == 14
    assert loaded["master_seed"] == 777
    assert loaded["rng_state"] == expected_rng_state
    assert (
        len(
            loaded[  # pyright: ignore[reportUnknownArgumentType]
                "final_population"
            ]
        )
        == 2
    )
    assert len(
        loaded[  # pyright: ignore[reportUnknownArgumentType]
            "hall_of_fame"
        ]
    ) == len(hall_of_fame)

    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = loaded["rng_state"]
    assert float(restored_rng.random()) == expected_next_draw


def test_handle_interrupt_saves_logbook_and_history_parquet(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "with_loggers"
    rng = np.random.default_rng(11)

    hall_of_fame = HallOfFame()
    population = _make_sample_population((1.0, -1.0))
    hall_of_fame.update(population)

    logbook = Logbook()
    logbook.record(1, _make_stats(1.5))
    logbook.record(3, _make_stats(2.5))

    history_logger = HistoryLogger()
    history_logger.record(1, hall_of_fame)
    history_logger.record(3, hall_of_fame)

    state = TrainingResult(
        final_population=population,
        logbook=logbook,
        history_logger=history_logger,
        hall_of_fame=hall_of_fame,
    )

    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=55,
    )
    saver.update_state(3, state)

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(2, None)

    assert exc_info.value.code == 0

    logbook_df = pl.read_parquet(
        output_directory / EMERGENCY_LOGBOOK_FILENAME  # type: ignore[arg-type]
    )
    history_df = pl.read_parquet(
        output_directory / EMERGENCY_HISTORY_FILENAME  # type: ignore[arg-type]
    )
    assert logbook_df["generation"].to_list() == [1, 3]
    assert history_df["generation"].to_list() == [1, 3]


def test_handle_interrupt_saves_only_logbook_when_history_is_none(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "logbook_only"
    rng = np.random.default_rng(13)
    logbook = Logbook()
    logbook.record(0, _make_stats(0.8))

    state = TrainingResult(
        final_population=[],
        logbook=logbook,
        history_logger=None,
        hall_of_fame=None,
    )

    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=17,
    )
    saver.update_state(0, state)

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(2, None)

    assert exc_info.value.code == 0
    assert (output_directory / EMERGENCY_LOGBOOK_FILENAME).is_file()
    assert not (output_directory / EMERGENCY_HISTORY_FILENAME).exists()


def test_handle_interrupt_saves_only_history_when_logbook_is_none(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "history_only"
    rng = np.random.default_rng(29)

    hall_of_fame = HallOfFame()
    population = _make_sample_population((1.0, -1.0))
    hall_of_fame.update(population)

    history_logger = HistoryLogger()
    history_logger.record(2, hall_of_fame)

    state = TrainingResult(
        final_population=population,
        logbook=None,
        history_logger=history_logger,
        hall_of_fame=hall_of_fame,
    )

    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=37,
    )
    saver.update_state(2, state)

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(2, None)

    assert exc_info.value.code == 0
    assert not (output_directory / EMERGENCY_LOGBOOK_FILENAME).exists()
    history_df = pl.read_parquet(
        output_directory / EMERGENCY_HISTORY_FILENAME  # type: ignore[arg-type]
    )
    assert history_df["generation"].to_list() == [2]


def test_update_state_overwrites_previous_generation_and_state(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "overwrite_test"
    rng = np.random.default_rng(99)

    first_logbook = Logbook()
    first_logbook.record(1, _make_stats(1.0))
    second_logbook = Logbook()
    second_logbook.record(5, _make_stats(4.0))

    first_state = TrainingResult(
        final_population=[],
        logbook=first_logbook,
        history_logger=None,
        hall_of_fame=None,
    )
    second_state = TrainingResult(
        final_population=[],
        logbook=second_logbook,
        history_logger=None,
        hall_of_fame=None,
    )

    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=101,
    )
    saver.update_state(1, first_state)
    saver.update_state(5, second_state)

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(2, None)

    assert exc_info.value.code == 0
    logbook_df = pl.read_parquet(
        output_directory / EMERGENCY_LOGBOOK_FILENAME  # type: ignore[arg-type]
    )
    assert logbook_df["generation"].to_list() == [5]

    loaded = _read_gzip_pickle(output_directory / EMERGENCY_RUNTIME_FILENAME)
    assert isinstance(loaded, dict)
    assert loaded["generation"] == 5


def test_handle_interrupt_rotates_existing_active_pointer(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "pointer_rotation"
    output_directory.mkdir(parents=True)
    pointer_path = output_directory / POINTER_FILENAME
    _ = pointer_path.write_text("runtime_gen_10.pkl.gz", encoding="utf-8")

    rng = np.random.default_rng(41)
    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=43,
    )
    saver.update_state(
        12,
        TrainingResult(
            final_population=[],
            logbook=None,
            history_logger=None,
            hall_of_fame=None,
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(2, None)

    assert exc_info.value.code == 0
    assert (
        pointer_path.read_text(encoding="utf-8") == EMERGENCY_RUNTIME_FILENAME
    )


def test_handle_interrupt_warns_but_exits_0_when_pointer_update_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_directory = tmp_path / "pointer_blocker"
    output_directory.mkdir(parents=True)
    (output_directory / POINTER_FILENAME).mkdir()

    rng = np.random.default_rng(31)
    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=73,
    )
    saver.update_state(
        2,
        TrainingResult(
            final_population=[],
            logbook=None,
            history_logger=None,
            hall_of_fame=None,
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        saver.handle_interrupt(2, None)

    assert exc_info.value.code == 0
    assert (output_directory / EMERGENCY_RUNTIME_FILENAME).is_file()
    captured = capsys.readouterr()
    out: str = captured.out
    assert "Warning: Could not rotate active checkpoint pointer" in out


def test_handle_interrupt_raises_when_output_path_is_a_file(
    tmp_path: Path,
) -> None:
    blocker_file = tmp_path / "not_a_directory"
    _ = blocker_file.write_bytes(b"blocking-file")

    rng = np.random.default_rng(19)
    saver = EmergencySaver(
        blocker_file,
        rng,
        _make_sample_algorithm(),
        master_seed=88,
    )
    saver.update_state(
        2,
        TrainingResult(
            final_population=[],
            logbook=None,
            history_logger=None,
            hall_of_fame=None,
        ),
    )

    with pytest.raises(OSError):
        saver.handle_interrupt(2, None)

    assert blocker_file.is_file()
    assert not (tmp_path / POINTER_FILENAME).exists()


def test_handle_interrupt_propagates_os_error_on_runtime_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "write_error_run"
    rng = np.random.default_rng(23)
    saver = EmergencySaver(
        output_directory, rng, _make_sample_algorithm(), master_seed=64
    )
    saver.update_state(
        4,
        TrainingResult(
            final_population=[],
            logbook=None,
            history_logger=None,
            hall_of_fame=None,
        ),
    )

    def _failing_gzip_open(*_args: object, **_kwargs: object) -> object:
        raise OSError("Simulated disk full during runtime save")

    monkeypatch.setattr(gzip, "open", _failing_gzip_open)

    with pytest.raises(OSError, match="Simulated disk full"):
        saver.handle_interrupt(2, None)

    assert not (output_directory / POINTER_FILENAME).exists()


def test_handle_interrupt_propagates_os_error_when_logbook_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "logbook_write_error"
    rng = np.random.default_rng(47)
    logbook = Logbook()
    logbook.record(1, _make_stats(1.2))

    def _failing_write_parquet(
        _self: object,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise OSError("Simulated disk failure writing parquet")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", _failing_write_parquet)

    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=53,
    )
    saver.update_state(
        1,
        TrainingResult(
            final_population=[],
            logbook=logbook,
            history_logger=None,
            hall_of_fame=None,
        ),
    )

    with pytest.raises(
        OSError,
        match="Simulated disk failure writing parquet",
    ):
        saver.handle_interrupt(2, None)

    assert not (output_directory / EMERGENCY_RUNTIME_FILENAME).exists()
    assert not (output_directory / POINTER_FILENAME).exists()


def test_handle_interrupt_propagates_pickling_error_on_unpicklable_payload(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "unpicklable_payload"
    rng = np.random.default_rng(59)

    def _unpicklable_local_callable() -> None:
        pass

    saver = EmergencySaver(
        output_directory,
        rng,
        _make_sample_algorithm(),
        master_seed=61,
    )
    # noinspection PyTypeChecker
    saver.update_state(
        6,
        TrainingResult(
            final_population=[_unpicklable_local_callable],  # pyright: ignore[reportArgumentType]
            logbook=None,
            history_logger=None,
            hall_of_fame=None,
        ),
    )

    with pytest.raises((pickle.PicklingError, AttributeError)):
        saver.handle_interrupt(2, None)

    assert not (output_directory / POINTER_FILENAME).exists()
