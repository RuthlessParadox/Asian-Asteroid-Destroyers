from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import polars as pl
import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    HallOfFame,
    HistoryLogger,
    clear_fitness_cache,
    clear_individual_cache,
    get_fitness_class,
    get_individual_class,
)


@pytest.fixture
def sample_hall_of_fame() -> HallOfFame:
    individual_class = get_individual_class((1.0, -1.0))

    hall_of_fame = HallOfFame(max_size=3)

    individual_1 = individual_class([0.1, 0.2, 0.3])
    individual_1.fitness.values = (10.0, -2.0)

    individual_2 = individual_class([0.4, 0.5, 0.6])
    individual_2.fitness.values = (5.0, -1.0)

    hall_of_fame.insert(individual_1)
    hall_of_fame.insert(individual_2)
    return hall_of_fame


@pytest.fixture(autouse=True)
def cleanup_caches(request: pytest.FixtureRequest) -> None:
    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


class _SnapshotDict(TypedDict):
    rank: int
    fitness: Sequence[float]
    genome: list[float]


def test_empty_logger_dataframe_schema() -> None:
    history_logger = HistoryLogger()
    dataframe = history_logger.dataframe

    assert isinstance(dataframe, pl.DataFrame)
    assert dataframe.height == 0
    assert "generation" in dataframe.columns
    assert "hall_of_fame" in dataframe.columns


def test_record_and_retrieve_dataframe(
    sample_hall_of_fame: HallOfFame,
) -> None:
    history_logger = HistoryLogger(flush_interval=5)
    history_logger.record(generation=0, hall_of_fame=sample_hall_of_fame)

    dataframe = history_logger.dataframe
    assert dataframe.height == 1
    assert dataframe["generation"][0] == 0

    hall_of_fame_list = cast(
        list[_SnapshotDict],
        dataframe["hall_of_fame"].item(0),
    )
    assert hall_of_fame_list is not None
    assert len(hall_of_fame_list) == 2

    first_rank: _SnapshotDict = hall_of_fame_list[0]
    assert first_rank is not None
    assert first_rank["rank"] == 1
    assert list(first_rank["fitness"]) == [10.0, -2.0]
    assert list(first_rank["genome"]) == [0.1, 0.2, 0.3]


def test_flush_interval_triggers_automatic_compile(
    sample_hall_of_fame: HallOfFame,
) -> None:
    history_logger = HistoryLogger(flush_interval=2)

    history_logger.record(generation=1, hall_of_fame=sample_hall_of_fame)
    history_logger.record(generation=2, hall_of_fame=sample_hall_of_fame)

    dataframe = history_logger.dataframe
    assert dataframe.height == 2
    assert list(dataframe["generation"]) == [1, 2]


def test_record_with_invalid_fitness(sample_hall_of_fame: HallOfFame) -> None:
    history_logger = HistoryLogger()

    for individual in sample_hall_of_fame:
        individual.fitness = get_fitness_class((1.0, -1.0))()

    history_logger.record(generation=10, hall_of_fame=sample_hall_of_fame)
    dataframe = history_logger.dataframe

    hof_list = cast(list[_SnapshotDict], dataframe["hall_of_fame"].item(0))
    assert hof_list is not None
    for snapshot in hof_list:
        assert list(snapshot["fitness"]) == []


def test_save_parquet_creates_file(
    sample_hall_of_fame: HallOfFame,
    tmp_path: Path,
) -> None:
    history_logger = HistoryLogger()
    history_logger.record(generation=1, hall_of_fame=sample_hall_of_fame)

    output_file = tmp_path / "history.parquet"
    history_logger.save_parquet(output_file)

    assert output_file.exists()

    loaded_dataframe = pl.read_parquet(output_file)
    assert loaded_dataframe.height == 1
    assert loaded_dataframe["generation"][0] == 1


def test_history_logger_string_representation(
    sample_hall_of_fame: HallOfFame,
) -> None:
    history_logger = HistoryLogger()
    history_logger.record(generation=5, hall_of_fame=sample_hall_of_fame)

    assert str(history_logger) == str(history_logger.dataframe)
