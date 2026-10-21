import sys
from typing import cast

from .individual import Fitness

_FITNESS_CLASS_CACHE: dict[tuple[float, ...], type[Fitness]] = {}


def get_fitness_class(weights: tuple[float, ...]) -> type[Fitness]:
    if weights in _FITNESS_CLASS_CACHE:
        return _FITNESS_CLASS_CACHE[weights]
    clean_weight_string = "_".join(
        str(weight).replace(".", "_").replace("-", "neg") for weight in weights
    )
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
