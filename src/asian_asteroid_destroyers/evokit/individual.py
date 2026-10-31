import sys
from collections.abc import Iterable
from typing import cast

from .cache_teardown import clear_dynamic_namespace_cache
from .fitness import Fitness, get_fitness_class, reconstruct_fitness_weights
from .weight_naming import get_clean_weight_string

DYNAMIC_INDIVIDUAL_PREFIX = "DynamicIndividual_"
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
    clean_weight_string = get_clean_weight_string(weights)
    class_name = f"{DYNAMIC_INDIVIDUAL_PREFIX}{clean_weight_string}"

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
    generic_cache = cast(
        dict[tuple[float, ...], type[object]], _INDIVIDUAL_CLASS_CACHE
    )
    clear_dynamic_namespace_cache(generic_cache, DYNAMIC_INDIVIDUAL_PREFIX)


def is_dynamic_individual_name(name: str) -> bool:
    return name.startswith(DYNAMIC_INDIVIDUAL_PREFIX)


def reconstruct_individual_weights(name: str) -> tuple[float, ...]:
    raw_weight_string = name[len(DYNAMIC_INDIVIDUAL_PREFIX) :]
    return reconstruct_fitness_weights(raw_weight_string)
