import pickle
import sys
from typing import cast

from typing_extensions import override

from asian_asteroid_destroyers.evokit import (
    get_individual_class,
    is_dynamic_fitness_name,
    is_dynamic_individual_name,
    reconstruct_fitness_weights,
    reconstruct_individual_weights,
)


class MainHydratingUnpickler(pickle.Unpickler):
    @override
    def find_class(self, module: str, name: str) -> object:
        if module == "__main__":
            main_module = sys.modules["__main__"]
            if not hasattr(main_module, name):
                if is_dynamic_individual_name(name):
                    weights = reconstruct_individual_weights(name)
                    _ = get_individual_class(weights)

                elif is_dynamic_fitness_name(name):
                    weights = reconstruct_fitness_weights(name)
                    _ = get_individual_class(weights)

        return cast(object, super().find_class(module, name))
