from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from asian_asteroid_destroyers.evokit import TrainingTimer

if TYPE_CHECKING:
    from pathlib import Path

LogValue = int | float
LogRow = Mapping[str, LogValue]


class TimeSaver:
    __slots__: tuple[str, ...] = ("_buffer", "_dataframes", "_flush_interval")
    _buffer: list[LogRow]
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

    def record(self, generation: int, training_time: float) -> None:
        self._buffer.append(
            {"generation": generation, "training_time": training_time}
        )

        if len(self._buffer) >= self._flush_interval:
            self._compile()

    def save_parquet(self, file_path: str | Path) -> None:
        self.dataframe.write_parquet(  # type: ignore[arg-type]
            file_path,
            compression="zstd",
            compression_level=3,
        )

    def hydrate(self, dataframe: pl.DataFrame) -> None:
        self._compile()
        self._dataframes = [dataframe]

    @staticmethod
    def _get_expected_schema() -> dict[str, pl.DataType]:
        return {"generation": pl.Int64(), "training_time": pl.Float64()}

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
