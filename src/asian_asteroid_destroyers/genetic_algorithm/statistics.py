from collections.abc import Callable

import numpy as np

from .individual import Individual
from .logbook import GenerationStats
from .types import Population


class Statistics:
    key: Callable[[Individual], tuple[float, ...]]
    index: int

    def __init__(
        self,
        key: Callable[[Individual], tuple[float, ...]] = lambda individual: (
            individual.fitness.values
        ),
        index: int = 0,
    ) -> None:
        self.key = key
        self.index = index

    def compile(self, population: Population) -> GenerationStats:
        if not population:
            raise ValueError(
                "Cannot compile statistics from an empty population."
            )
        try:
            data = np.fromiter(
                (
                    self.key(individual)[self.index]
                    for individual in population
                ),
                dtype=np.float64,
                count=len(population),
            )
        except IndexError:
            raise IndexError(
                f"Target index {self.index} is out of bounds for "
                + "one or more individuals in the population."
            )
        return {
            "max_fitness": float(np.max(data)),
            "min_fitness": float(np.min(data)),
            "mean_fitness": float(np.mean(data)),
            "median_fitness": float(np.median(data)),
            "std_fitness": float(np.std(data)),
        }
