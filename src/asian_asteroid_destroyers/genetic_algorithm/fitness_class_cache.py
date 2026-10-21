import math
import sys
from typing import cast

from .fitness import Fitness

_FITNESS_CLASS_CACHE: dict[tuple[float, ...], type[Fitness]] = {}


def get_fitness_class(weights: tuple[float, ...]) -> type[Fitness]:
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError(
            "Fitness weights must be finite numbers (e.g., 1.0, -1.0)."
        )
    if weights in _FITNESS_CLASS_CACHE:
        return _FITNESS_CLASS_CACHE[weights]
    clean_weight_string = Fitness.get_clean_weight_string(weights)
    class_name = f"DynamicFitness_{clean_weight_string}"
    new_class = cast(
        type[Fitness],
        type(class_name, (Fitness,), {"weights": weights}),
    )
    current_module = sys.modules[__name__]
    setattr(current_module, class_name, new_class)
    new_class.__module__ = __name__
    _FITNESS_CLASS_CACHE[weights] = new_class
    return new_class


def clear_fitness_cache() -> None:
    current_module = sys.modules[__name__]

    for weight_key in list(_FITNESS_CLASS_CACHE.keys()):
        clean_weight_string = Fitness.get_clean_weight_string(weight_key)
        class_name = f"DynamicFitness_{clean_weight_string}"
        if hasattr(current_module, class_name):
            delattr(current_module, class_name)

    _FITNESS_CLASS_CACHE.clear()
