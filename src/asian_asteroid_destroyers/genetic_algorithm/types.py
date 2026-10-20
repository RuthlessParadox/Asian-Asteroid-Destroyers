from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np
from deap import base
from typing_extensions import override


class TypedFitness(base.Fitness):
    weights: tuple[float, ...] = ()
    values: tuple[float, ...]

    def __init__(self, values: tuple[float, ...] | None = None) -> None:
        super().__init__()
        if values is not None:
            self.values = values

    @override
    def __setattr__(self, name: str, value: object) -> None:
        if name == "values" and isinstance(value, tuple):
            tuple_length = value.__len__()
            if tuple_length != len(self.weights):
                raise ValueError(
                    f"Mismatched dimensions: Assigned {tuple_length} values, "
                    + f"but fitness weights expect {len(self.weights)}."
                )

        super().__setattr__(name, value)


class Individual(list[float]):
    fitness: TypedFitness

    def __init__(self, iterable: Iterable[float] | None = None) -> None:
        if iterable is not None:
            super().__init__(iterable)
        else:
            super().__init__()
        if not hasattr(self, "fitness"):
            self.fitness = TypedFitness()


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
