from collections.abc import Sequence

import polars as pl
from typing_extensions import override


class Logbook:
    header: Sequence[str]
    _records: list[dict[str, int | float]]

    def __init__(self, header: Sequence[str] | None = None) -> None:
        self.header = header if header is not None else []
        self._records = []

    def record(self, **kwargs: float) -> None:
        self._records.append(kwargs)

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(self._records)

    @override
    def __str__(self) -> str:
        if not self._records:
            return "Empty Logbook"
        return str(self.to_dataframe())
