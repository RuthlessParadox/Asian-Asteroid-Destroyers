from collections.abc import Callable, Iterable, Sequence
from typing import TypedDict

import numpy as np

from .individual import Gene, Individual

__all__ = [
    "Gene",
    "Individual",
    "Population",
    "GenerationStats",
    "IntVector",
    "IntMatrix",
    "FloatVector",
    "FloatMatrix",
    "BoolVector",
    "BoolMatrix",
    "IndividualGenerator",
    "PopulationGenerator",
    "CloneMethod",
    "MapMethod",
    "FitnessEvaluator",
    "CrossoverOperator",
    "MutationOperator",
    "SelectionOperator",
]

Population = Sequence[Individual]


class GenerationStats(TypedDict):
    max_fitness: float
    min_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float


IntVector = np.ndarray[tuple[int], np.dtype[np.int64]]
IntMatrix = np.ndarray[tuple[int, int], np.dtype[np.int64]]
FloatVector = np.ndarray[tuple[int], np.dtype[np.float64]]
FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
BoolVector = np.ndarray[tuple[int], np.dtype[np.bool]]
BoolMatrix = np.ndarray[tuple[int, int], np.dtype[np.bool]]

IndividualGenerator = Callable[[], Individual]
PopulationGenerator = Callable[[int], Population]
CloneMethod = Callable[[Individual], Individual]
MapMethod = Callable[
    [Callable[[Individual], tuple[float, ...]], list[Individual]],
    Iterable[tuple[float, ...]],
]

FitnessEvaluator = Callable[[Individual, int], tuple[float, ...]]
CrossoverOperator = Callable[
    [Individual, Individual], tuple[Individual, Individual]
]
MutationOperator = Callable[[Individual], tuple[Individual]]
SelectionOperator = Callable[[Population, int], Population]
