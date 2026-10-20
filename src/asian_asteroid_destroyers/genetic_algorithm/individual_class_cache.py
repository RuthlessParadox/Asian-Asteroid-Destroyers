import sys
from collections.abc import Iterable
from typing import cast

from .fitness_class_cache import get_fitness_class
from .individuals import Individual

_INDIVIDUAL_CLASS_CACHE: dict[tuple[float, ...], type[Individual]] = {}


def get_individual_class(weights: tuple[float, ...]) -> type[Individual]:
    if weights in _INDIVIDUAL_CLASS_CACHE:
        return _INDIVIDUAL_CLASS_CACHE[weights]
    fitness_type = get_fitness_class(weights)
    clean_weight_string = "_".join(
        str(weight).replace(".", "_").replace("-", "neg") for weight in weights
    )
    class_name = f"DynamicIndividual_{clean_weight_string}"

    def individual_init(
        self: Individual,
        iterable: Iterable[float] | None = None,
    ) -> None:
        super(Individual, self).__init__(iterable or [])
        self.fitness = fitness_type()

    new_class = cast(
        type[Individual],
        type(class_name, (Individual,), {"__init__": individual_init}),
    )
    current_module = sys.modules[__name__]
    setattr(current_module, class_name, new_class)
    new_class.__module__ = __name__
    _INDIVIDUAL_CLASS_CACHE[weights] = new_class
    return new_class
