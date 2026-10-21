import sys
from typing import cast

from .individual import TypedFitness

_FITNESS_CLASS_CACHE: dict[tuple[float, ...], type[TypedFitness]] = {}


def get_fitness_class(weights: tuple[float, ...]) -> type[TypedFitness]:
    if weights in _FITNESS_CLASS_CACHE:
        return _FITNESS_CLASS_CACHE[weights]
    clean_weight_string = "_".join(
        str(weight).replace(".", "_").replace("-", "neg") for weight in weights
    )
    class_name = f"DynamicFitness_{clean_weight_string}"
    new_class = cast(
        type[TypedFitness],
        type(class_name, (TypedFitness,), {"weights": weights}),
    )
    current_module = sys.modules[__name__]
    setattr(current_module, class_name, new_class)
    new_class.__module__ = __name__
    _FITNESS_CLASS_CACHE[weights] = new_class
    return new_class
