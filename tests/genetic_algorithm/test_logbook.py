import os
from pathlib import Path

import polars as pl
import pytest

from asian_asteroid_destroyers.genetic_algorithm.logbook import (
    GenerationStats,
    Logbook,
)


@pytest.fixture
def valid_stats() -> GenerationStats:
    return {
        "max_fitness": 10.0,
        "min_fitness": 1.0,
        "mean_fitness": 5.5,
        "median_fitness": 5.0,
        "std_fitness": 2.5,
    }


@pytest.fixture
def temporary_file_path(tmp_path: Path) -> str:
    file_path = tmp_path / "test_output.parquet"
    return str(file_path)


def test_empty_logbook_returns_empty_dataframe_with_correct_schema() -> None:
    logbook = Logbook()
    dataframe = logbook.dataframe

    assert isinstance(dataframe, pl.DataFrame)
    assert dataframe.height == 0
    assert dataframe.schema == {
        "generation": pl.Int32(),
        "max_fitness": pl.Float32(),
        "min_fitness": pl.Float32(),
        "mean_fitness": pl.Float32(),
        "median_fitness": pl.Float32(),
        "std_fitness": pl.Float32(),
    }


def test_record_adds_to_dataframe(valid_stats: GenerationStats) -> None:
    logbook = Logbook(flush_interval=5)
    logbook.record(generation=1, stats=valid_stats)

    dataframe = logbook.dataframe
    assert dataframe.height == 1
    assert dataframe["generation"][0] == 1
    assert dataframe["max_fitness"][0] == 10.0


def test_flush_interval_triggers_compilation(
    valid_stats: GenerationStats,
) -> None:
    logbook = Logbook(flush_interval=2)

    logbook.record(generation=1, stats=valid_stats)
    logbook.record(generation=2, stats=valid_stats)

    dataframe = logbook.dataframe
    assert dataframe.height == 2


def test_dataframe_concatenates_multiple_flushed_chunks(
    valid_stats: GenerationStats,
) -> None:
    logbook = Logbook(flush_interval=1)

    logbook.record(generation=1, stats=valid_stats)
    logbook.record(generation=2, stats=valid_stats)

    dataframe = logbook.dataframe
    assert dataframe.height == 2
    assert list(dataframe["generation"]) == [1, 2]


def test_save_parquet_creates_valid_file(
    valid_stats: GenerationStats,
    temporary_file_path: str,
) -> None:
    logbook = Logbook()
    logbook.record(generation=1, stats=valid_stats)
    logbook.save_parquet(temporary_file_path)

    assert os.path.exists(temporary_file_path)

    loaded_dataframe = pl.read_parquet(temporary_file_path)
    assert loaded_dataframe.height == 1
    assert loaded_dataframe["generation"][0] == 1


def test_save_parquet_overwrites_existing_file(
    valid_stats: GenerationStats,
    temporary_file_path: str,
) -> None:
    logbook = Logbook()

    logbook.record(generation=1, stats=valid_stats)
    logbook.save_parquet(temporary_file_path)

    logbook.record(generation=2, stats=valid_stats)
    logbook.save_parquet(temporary_file_path)

    loaded_dataframe = pl.read_parquet(temporary_file_path)

    assert loaded_dataframe.height == 2
    assert list(loaded_dataframe["generation"]) == [1, 2]


def test_string_representation_matches_dataframe_string(
    valid_stats: GenerationStats,
) -> None:
    logbook = Logbook()
    logbook.record(generation=1, stats=valid_stats)

    assert str(logbook) == str(logbook.dataframe)
