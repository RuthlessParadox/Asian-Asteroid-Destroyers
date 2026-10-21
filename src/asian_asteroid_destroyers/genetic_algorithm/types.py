from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np

from .individual import Individual

_Shape1D = TypeVar("_Shape1D", bound=tuple[int])
_Shape2D = TypeVar("_Shape2D", bound=tuple[int, int])

Population = list[Individual]
CloneMethod = Callable[[Individual], Individual]
MapMethod = Callable[
    [Callable[[Individual], tuple[float, ...]], list[Individual]],
    Iterable[tuple[float, ...]],
]
IndividualGenerator = Callable[[], Individual]
PopulationGenerator = Callable[[int], Population]
FitnessEvaluator = Callable[[Individual], tuple[float, ...]]
CrossoverOperator = Callable[
    [Individual, Individual], tuple[Individual, Individual]
]
MutationOperator = Callable[[Individual], tuple[Individual]]
SelectionOperator = Callable[[Population, int], Population]
FloatVector = np.ndarray[_Shape1D, np.dtype[np.float64]]
FloatMatrix = np.ndarray[_Shape2D, np.dtype[np.float64]]
