import pickle
import sys
from typing import cast

from typing_extensions import override

from asian_asteroid_destroyers.genetic_algorithm import get_individual_class


class MainHydratingUnpickler(pickle.Unpickler):
    @override
    def find_class(self, module: str, name: str) -> object:
        if module == "__main__" and (
            name.startswith("DynamicIndividual_")
            or name.startswith("DynamicFitness_")
        ):
            main_module = sys.modules["__main__"]
            if not hasattr(main_module, name):
                prefix = (
                    "DynamicIndividual_"
                    if name.startswith("DynamicIndividual_")
                    else "DynamicFitness_"
                )
                raw_weight_string = name[len(prefix) :]

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

                _ = get_individual_class(tuple(weights))

        return cast(object, super().find_class(module, name))
