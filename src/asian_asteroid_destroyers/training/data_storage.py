"""Data Storage Module.

This module provides functions to store training data locally and onto the
disc.
"""

from datetime import datetime
from typing import get_args, get_origin, get_type_hints, NamedTuple, TypeAlias

import polars as pl

_MappableType: TypeAlias = type[int] | type[float] | type[str] | type[bool]
"""Native Python types that can be mapped to Polars types."""
_MappedType: TypeAlias = (
    pl.List
    | type[pl.Int32]
    | type[pl.Float32]
    | type[pl.String]
    | type[pl.Boolean]
    | type[pl.Unknown]
)
"""Polars types that can be mapped to from a _MappableType."""


class GenerationData(NamedTuple):
    """A collection of data representing the best solution in a generation.

    Attributes
    ----------
    generation : int
        The generation.
    fitness : float
        The best solution's fitness.
    genome : list[float]
        The best solution's genes.
    number_of_genes : int
        The number of genes in the best solution's genome.
    population_size : int
        The number of solutions in the generation.
    cxpb : float
        The generation's crossover probability.
    mutpb : float
        The generation's mutation probability.
    """
    generation: int
    fitness: float
    genome: list[float]
    number_of_genes: int
    population_size: int
    cxpb: float
    mutpb: float


def build_best_solution_lazy_frame() -> pl.LazyFrame:
    """Builds a lazy frame to store the best solutions in a training session.
    The frame's schema is based on the NamedTuple GenerationData.

    Returns
    -------
    pl.LazyFrame
        The lazy frame to store a training session's best solutions.
    """
    type_hints: dict[str, _MappableType] = get_type_hints(GenerationData)
    schema = {
        field: _map_py_type_to_polars_type(py_type)
        for field, py_type in type_hints.items()
    }
    return pl.LazyFrame(schema)


def _map_py_type_to_polars_type(py_type: _MappableType) -> _MappedType:
    """Maps a native Python type to a Polars type.

    Parameters
    ----------
    py_type : _MappableType
        The Python type to map. The type must be mappable.

    Returns
    -------
    _MappedType
        The Polars type that was mapped to.
        This can also be a pl.List or a pl.Unknown.
    """
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
    """Adds a row of generation data to a training session's best solutions.

    Parameters
    ----------
    best_solutions : pl.LazyFrame
        The lazy frame storing the training session's best solutions.
    data : GenerationData
        The row of generation data to add.

    Returns
    -------
    pl.LazyFrame
        The updated lazy frame with the added data.
    """
    new_row = pl.DataFrame([data]).lazy()
    return pl.concat([best_solutions, new_row], how="vertical")


def save_best_solutions(best_solutions: pl.LazyFrame, timestamp: str) -> None:
    """Saves a training session's best solutions to a parquet file.

    Parameters
    ----------
    best_solutions : pl.LazyFrame
        The lazy frame storing the training session's best solutions.
    timestamp : str
        The timestamp to use when naming the parquet file.

    Returns
    -------
    None
    """
    best_solutions.sink_parquet(
        _build_best_solutions_file_path(timestamp),
        compression="zstd",
        compression_level=3
    )


def _build_best_solutions_file_path(timestamp: str) -> str:
    """Builds a file path for a parquet file storing a training session's best
    solutions. Files are named "best_solutions_{timestamp}.parquet".

    Parameters
    ----------
    timestamp : str
        The timestamp to use when naming the parquet file.

    Returns
    -------
    str
        The built file path.
    """
    return f"best_solutions_{timestamp}.parquet"


def build_timestamp() -> str:
    """Builds a timestamp for the current date and time.

    Returns
    -------
    str
        The current timestamp in the form YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")