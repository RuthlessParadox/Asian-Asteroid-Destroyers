from collections.abc import Callable, Iterable, Sequence
from typing import TypedDict, TypeVar

import numpy as np

from .individual import Gene, Individual

__all__ = [
    "Gene",
    "Individual",
    "Population",
    "GenerationStats",
    "FloatVector",
    "FloatMatrix",
    "IndividualGenerator",
    "PopulationGenerator",
    "CloneMethod",
    "MapMethod",
    "FitnessEvaluator",
    "CrossoverOperator",
    "MutationOperator",
    "SelectionOperator",
]

_Shape1D = TypeVar("_Shape1D", bound=tuple[int])
_Shape2D = TypeVar("_Shape2D", bound=tuple[int, int])

Population = Sequence[Individual]


class GenerationStats(TypedDict):
    max_fitness: float
    min_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float


FloatVector = np.ndarray[_Shape1D, np.dtype[np.float64]]
FloatMatrix = np.ndarray[_Shape2D, np.dtype[np.float64]]

IndividualGenerator = Callable[[], Individual]
PopulationGenerator = Callable[[int], Population]
CloneMethod = Callable[[Individual], Individual]
MapMethod = Callable[
    [Callable[[Individual], tuple[float, ...]], list[Individual]],
    Iterable[tuple[float, ...]],
]

FitnessEvaluator = Callable[[Individual], tuple[float, ...]]
CrossoverOperator = Callable[
    [Individual, Individual], tuple[Individual, Individual]
]
MutationOperator = Callable[[Individual], tuple[Individual]]
SelectionOperator = Callable[[Population, int], Population]
