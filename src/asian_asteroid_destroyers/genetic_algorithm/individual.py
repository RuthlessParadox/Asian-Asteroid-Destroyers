import sys
from collections.abc import Iterable
from typing import cast

from .fitness import Fitness, get_fitness_class

Gene = float


class Individual(list[Gene]):
    fitness: Fitness
    fitness_type: type[Fitness]

    def __init__(self, iterable: Iterable[Gene] | None = None) -> None:
        if iterable is not None:
            super().__init__(iterable)
        else:
            super().__init__()
        if not hasattr(self, "fitness"):
            self.fitness = Fitness()


_INDIVIDUAL_CLASS_CACHE: dict[tuple[float, ...], type[Individual]] = {}


def _dynamic_individual_init(
    self: Individual, iterable: Iterable[Gene] | None = None
) -> None:
    super(Individual, self).__init__(iterable or [])
    self.fitness = self.fitness_type()


def get_individual_class(weights: tuple[float, ...]) -> type[Individual]:
    if weights in _INDIVIDUAL_CLASS_CACHE:
        return _INDIVIDUAL_CLASS_CACHE[weights]

    fitness_class = get_fitness_class(weights)
    clean_weight_string = Fitness.get_clean_weight_string(weights)
    class_name = f"DynamicIndividual_{clean_weight_string}"

    class_dict = {
        "__init__": _dynamic_individual_init,
        "fitness_type": fitness_class,
    }

    new_class = cast(
        type[Individual],
        type(class_name, (Individual,), class_dict),
    )

    main_module = sys.modules["__main__"]
    setattr(main_module, class_name, new_class)
    new_class.__module__ = "__main__"
    _INDIVIDUAL_CLASS_CACHE[weights] = new_class
    return new_class


def clear_individual_cache() -> None:
    main_module = sys.modules["__main__"]

    for weight_key in list(_INDIVIDUAL_CLASS_CACHE.keys()):
        clean_weight_string = Fitness.get_clean_weight_string(weight_key)
        class_name = f"DynamicIndividual_{clean_weight_string}"
        if hasattr(main_module, class_name):
            delattr(main_module, class_name)

    _INDIVIDUAL_CLASS_CACHE.clear()
