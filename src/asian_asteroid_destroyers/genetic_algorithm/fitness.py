import math
import sys
from typing import cast

from deap import base
from typing_extensions import override

DYNAMIC_FITNESS_PREFIX = "DynamicFitness_"


class Fitness(base.Fitness):
    weights: tuple[float, ...] = ()
    values: tuple[float, ...]

    def __init__(self, values: tuple[float, ...] | None = None) -> None:
        super().__init__()
        if values is not None:
            self.values = values

    @property
    @override
    def valid(self) -> bool:
        return bool(self.values)

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

    @staticmethod
    def get_clean_weight_string(weights: tuple[float, ...]) -> str:
        return "_".join(
            str(weight).replace(".", "_").replace("-", "neg")
            for weight in weights
        )


_FITNESS_CLASS_CACHE: dict[tuple[float, ...], type[Fitness]] = {}


def get_fitness_class(weights: tuple[float, ...]) -> type[Fitness]:
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError(
            "Fitness weights must be finite numbers (e.g., 1.0, -1.0)."
        )
    if weights in _FITNESS_CLASS_CACHE:
        return _FITNESS_CLASS_CACHE[weights]
    clean_weight_string = Fitness.get_clean_weight_string(weights)
    class_name = f"{DYNAMIC_FITNESS_PREFIX}{clean_weight_string}"
    new_class = cast(
        type[Fitness],
        type(class_name, (Fitness,), {"weights": weights}),
    )

    main_module = sys.modules["__main__"]
    setattr(main_module, class_name, new_class)
    new_class.__module__ = "__main__"

    _FITNESS_CLASS_CACHE[weights] = new_class
    return new_class


def clear_fitness_cache() -> None:
    main_module = sys.modules["__main__"]

    for weight_key in list(_FITNESS_CLASS_CACHE.keys()):
        clean_weight_string = Fitness.get_clean_weight_string(weight_key)
        class_name = f"{DYNAMIC_FITNESS_PREFIX}{clean_weight_string}"
        if hasattr(main_module, class_name):
            delattr(main_module, class_name)

    _FITNESS_CLASS_CACHE.clear()


def is_dynamic_fitness_name(name: str) -> bool:
    return name.startswith(DYNAMIC_FITNESS_PREFIX)


def reconstruct_fitness_weights(name: str) -> tuple[float, ...]:
    raw_weight_string = (
        name[len(DYNAMIC_FITNESS_PREFIX) :]
        if name.startswith(DYNAMIC_FITNESS_PREFIX)
        else name
    )

    parts = raw_weight_string.split("_")
    weights: list[float] = []
    i = 0
    while i < len(parts):
        value_str = parts[i]
        if value_str.startswith("neg"):
            value_str = "-" + value_str[3:]
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            value_str = f"{value_str}.{parts[i + 1]}"
            i += 1
        try:
            weights.append(float(value_str))
        except ValueError:
            pass
        i += 1
    return tuple(weights)
