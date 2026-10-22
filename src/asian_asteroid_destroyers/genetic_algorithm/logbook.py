from collections.abc import Mapping
from typing import cast

import polars as pl
from typing_extensions import override

from .types import GenerationStats

LogValue = int | float
LogRow = dict[str, LogValue]

_TYPE_MAPPING: dict[type, pl.DataType] = {
    int: pl.Int64(),
    float: pl.Float64(),
    str: pl.String(),
    bool: pl.Boolean(),
}


class Logbook:
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

    def record(self, generation: int, stats: GenerationStats) -> None:
        stats_map = cast(Mapping[str, LogValue], stats)
        record = {
            "generation": generation,
            **stats_map,
        }
        self._buffer.append(record)

        if len(self._buffer) >= self._flush_interval:
            self._compile()

    def save_parquet(self, file_path: str) -> None:
        self.dataframe.write_parquet(
            file_path,
            compression="zstd",
            compression_level=3,
        )

    @staticmethod
    def _get_expected_schema() -> dict[str, pl.DataType]:
        schema: dict[str, pl.DataType] = {"generation": pl.Int32()}

        annotations: dict[str, type] = GenerationStats.__annotations__
        for field_name, python_type in annotations.items():
            polars_type = _TYPE_MAPPING.get(python_type)
            if polars_type is None:
                raise TypeError(
                    f"Unsupported type '{python_type.__name__}' detected on "
                    + f"GenerationStats field '{field_name}'."
                )
            schema[field_name] = polars_type

        return schema

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
