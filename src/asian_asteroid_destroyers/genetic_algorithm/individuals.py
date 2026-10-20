from collections.abc import Iterable

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
