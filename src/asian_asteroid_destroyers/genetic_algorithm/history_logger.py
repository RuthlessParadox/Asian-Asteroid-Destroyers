from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import polars as pl
from typing_extensions import override

from .hall_of_fame import HallOfFame
from .types import Gene


class _IndividualSnapshot(TypedDict):
    rank: int
    fitness: Sequence[float]
    genome: list[Gene]


class _HistoryRow(TypedDict):
    generation: int
    hall_of_fame: list[_IndividualSnapshot]


class HistoryLogger:
    _buffer: list[_HistoryRow]
    _dataframes: list[pl.DataFrame]
    _flush_interval: int

    def __init__(self, flush_interval: int = 100) -> None:
        self._buffer = []
        self._dataframes = []
        self._flush_interval = flush_interval

    @property
    def dataframe(self) -> pl.DataFrame:
        self._compile()

        if not self._dataframes:
            return pl.DataFrame(schema=self._get_expected_schema())

        return pl.concat(self._dataframes, how="vertical")

    def record(self, generation: int, hall_of_fame: HallOfFame) -> None:
        hall_of_fame_records: list[_IndividualSnapshot] = []

        for rank, individual in enumerate(hall_of_fame, start=1):
            fitness_values = (
                individual.fitness.values if individual.fitness.valid else ()
            )
            hall_of_fame_records.append(
                {
                    "rank": rank,
                    "fitness": fitness_values,
                    "genome": individual[:],
                }
            )

        record: _HistoryRow = {
            "generation": generation,
            "hall_of_fame": hall_of_fame_records,
        }
        self._buffer.append(record)

        if len(self._buffer) >= self._flush_interval:
            self._compile()

    def save_parquet(self, file_path: str | Path) -> None:
        self.dataframe.write_parquet(
            file_path,
            compression="zstd",
            compression_level=3,
        )

    def hydrate(self, dataframe: pl.DataFrame) -> None:
        self._compile()
        self._dataframes = [dataframe]

    @staticmethod
    def _get_expected_schema() -> dict[str, pl.DataType]:
        return {
            "generation": pl.Int64(),
            "hall_of_fame": pl.List(
                pl.Struct(
                    [
                        pl.Field("rank", pl.Int64()),
                        pl.Field("fitness", pl.List(pl.Float64())),
                        pl.Field("genome", pl.List(pl.Float64())),
                    ]
                )
            ),
        }

    def _compile(self) -> None:
        if not self._buffer:
            return

        new_chunk = pl.DataFrame(
            self._buffer,
            schema=self._get_expected_schema(),
        )

        self._dataframes.append(new_chunk)
        self._buffer.clear()

    @override
    def __str__(self) -> str:
        return str(self.dataframe)
