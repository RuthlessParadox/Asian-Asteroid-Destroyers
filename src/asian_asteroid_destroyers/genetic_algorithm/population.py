import copy

from .fitness_class_cache import get_fitness_class
from .types import Population


def reweight_population(
    population: Population,
    new_weights: tuple[float, ...],
) -> None:
    new_fitness_type = get_fitness_class(new_weights)
    empty_prototype = new_fitness_type()

    for individual in population:
        fitness = individual.fitness
        raw_values: tuple[float, ...] | None = None

        if getattr(fitness, "valid", False):
            try:
                raw_values = fitness.values
            except AttributeError:
                pass

        if raw_values is not None:
            new_fitness = new_fitness_type()
            new_fitness.values = raw_values
            individual.fitness = new_fitness
        else:
            individual.fitness = copy.copy(empty_prototype)
