# -*- coding: utf-8 -*-
# Authors: Isaac Zhou
# File Name: data_storage.py
# Copyright © 2026 DigiPen Institute of Technology. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from datetime import datetime
from typing import get_args, get_origin, get_type_hints, NamedTuple, TypeAlias

import polars as pl

_MappableType: TypeAlias = type[int] | type[float] | type[str] | type[bool]


class GenerationData(NamedTuple):
    generation: int
    fitness: float
    genome: list[float]
    number_of_genes: int
    population_size: int
    cxpb: float
    mutpb: float


def build_best_solution_lazyframe() -> pl.LazyFrame:
    type_hints: dict[str, _MappableType] = get_type_hints(GenerationData)
    schema = {
        field: _map_py_type_to_polars_type(py_type)
        for field, py_type in type_hints.items()
    }
    return pl.LazyFrame(schema)


def _map_py_type_to_polars_type(py_type: _MappableType):
    type_mapping = {
        int: pl.Int32,
        float: pl.Float32,
        str: pl.String,
        bool: pl.Boolean,
    }

    # Check if it's a generic type hint like list[float]
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is list:
        # Recursively find the inner Polars type (defaulting to Float32 if unknown)
        inner_py_type = args[0] if args else float
        inner_pl_type = type_mapping.get(inner_py_type, pl.Float32)
        return pl.List(inner_pl_type)

    # Standard primitive lookup
    return type_mapping.get(py_type, pl.Unknown)


def add_generation(
    best_solutions: pl.LazyFrame,
    data: GenerationData,
) -> pl.LazyFrame:
    new_row = pl.DataFrame([data]).lazy()
    return pl.concat([best_solutions, new_row], how="vertical")


def save_best_solutions(best_solutions: pl.LazyFrame, timestamp: str) -> None:
    best_solutions.sink_parquet(
        _build_best_solutions_file_path(timestamp),
        compression="zstd",
        compression_level=3
    )


def _build_best_solutions_file_path(timestamp: str) -> str:
    return f"best_solutions_{timestamp}.parquet"


def build_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")