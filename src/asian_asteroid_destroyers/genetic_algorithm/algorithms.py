from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from .hall_of_fame import HallOfFame
from .history_logger import HistoryLogger
from .logbook import Logbook
from .statistics import Statistics
from .toolbox import Toolbox
from .types import Population


class TrainingResult(NamedTuple):
    final_population: Population
    logbook: Logbook | None
    history_logger: HistoryLogger | None
    hall_of_fame: HallOfFame | None


def run_simple_evolution(
    toolbox: Toolbox,
    cx_pb: float,
    mut_pb: float,
    generations: int,
    *,
    seed: int | np.random.Generator | None = None,
    hall_of_fame: HallOfFame | None = None,
    statistics: Statistics | None = None,
    logbook: Logbook | None = None,
    history_logger: HistoryLogger | None = None,
    checkpoint_callback: Callable[[int, TrainingResult], None] | None = None,
) -> TrainingResult:
    if not (0.0 <= cx_pb <= 1.0):
        raise ValueError(
            "Crossover probability 'cx_pb' must be between 0.0 and 1.0."
            + f"Got: {cx_pb}"
        )
    if not (0.0 <= mut_pb <= 1.0):
        raise ValueError(
            "Mutation probability 'mut_pb' must be between 0.0 and 1.0."
            + f"Got: {mut_pb}"
        )
    if generations < 0:
        raise ValueError(
            "Number of generations 'n_gen' must be non-negative."
            + f"Got: {generations}"
        )

    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )

    population: Population = list(toolbox.population(generations))

    invalid_individuals = [
        individual for individual in population if not individual.fitness.valid
    ]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
    for individual, fitness in zip(invalid_individuals, fitnesses):
        individual.fitness.values = fitness

    if hall_of_fame is not None:
        hall_of_fame.update(population)

    if statistics is not None and logbook is not None:
        logbook.record(0, statistics.compile(population))

    if history_logger is not None and hall_of_fame is not None:
        history_logger.record(0, hall_of_fame)

    for generation in range(1, generations + 1):
        offspring = list(toolbox.select(population, len(population)))

        offspring = [toolbox.clone(individual) for individual in offspring]

        for i in range(1, len(offspring), 2):
            if rng.random() < cx_pb:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1],
                    offspring[i],
                )
                try:
                    del offspring[i - 1].fitness.values
                except AttributeError:
                    pass
                try:
                    del offspring[i].fitness.values
                except AttributeError:
                    pass

        for i in range(len(offspring)):
            if rng.random() < mut_pb:
                (offspring[i],) = toolbox.mutate(offspring[i])
                try:
                    del offspring[i].fitness.values
                except AttributeError:
                    pass

        invalid_individuals = [
            individual
            for individual in offspring
            if not individual.fitness.valid
        ]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for individual, fitness in zip(invalid_individuals, fitnesses):
            individual.fitness.values = fitness

        population = offspring

        if hall_of_fame is not None:
            hall_of_fame.update(population)

        if statistics is not None and logbook is not None:
            logbook.record(generation, statistics.compile(population))

        if history_logger is not None and hall_of_fame is not None:
            history_logger.record(generation, hall_of_fame)

        if checkpoint_callback is not None:
            checkpoint_callback(
                generation,
                TrainingResult(
                    final_population=population,
                    logbook=logbook,
                    history_logger=history_logger,
                    hall_of_fame=hall_of_fame,
                ),
            )

    return TrainingResult(
        final_population=population,
        logbook=logbook,
        history_logger=history_logger,
        hall_of_fame=hall_of_fame,
    )
