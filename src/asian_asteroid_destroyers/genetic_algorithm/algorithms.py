from collections.abc import Callable, MutableSequence
from typing import NamedTuple

import numpy as np

from .hall_of_fame import HallOfFame
from .history_logger import HistoryLogger
from .individual import Individual
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
    population_size: int,
    *,
    seed: int | np.random.Generator | None = None,
    hall_of_fame: HallOfFame | None = None,
    statistics: Statistics | None = None,
    logbook: Logbook | None = None,
    history_logger: HistoryLogger | None = None,
    checkpoint_callback: Callable[[int, TrainingResult], None] | None = None,
) -> TrainingResult:
    _validate_parameters(cx_pb, mut_pb, generations, population_size)

    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )

    population: Population = list(toolbox.population(population_size))

    _evaluate_invalid_fitness(toolbox, population)
    _record_generation_metrics(
        generation=0,
        population=population,
        hall_of_fame=hall_of_fame,
        statistics=statistics,
        logbook=logbook,
        history_logger=history_logger,
        checkpoint_callback=checkpoint_callback,
    )

    for generation in range(1, generations + 1):
        offspring = list(toolbox.select(population, len(population)))
        offspring = [toolbox.clone(individual) for individual in offspring]

        _apply_crossover(toolbox, offspring, cx_pb, rng)
        _apply_mutation(toolbox, offspring, mut_pb, rng)

        _evaluate_invalid_fitness(toolbox, offspring)
        population = offspring

        _record_generation_metrics(
            generation=generation,
            population=population,
            hall_of_fame=hall_of_fame,
            statistics=statistics,
            logbook=logbook,
            history_logger=history_logger,
            checkpoint_callback=checkpoint_callback,
        )

    return TrainingResult(
        final_population=population,
        logbook=logbook,
        history_logger=history_logger,
        hall_of_fame=hall_of_fame,
    )


def _validate_parameters(
    cx_pb: float,
    mut_pb: float,
    generations: int,
    population_size: int,
) -> None:
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
    if population_size <= 0:
        raise ValueError(
            "Population size 'population_size' must be a positive integer."
            + f"Got: {population_size}"
        )


def _evaluate_invalid_fitness(
    toolbox: Toolbox,
    population: Population,
) -> None:
    invalid_individuals = [
        individual for individual in population if not individual.fitness.valid
    ]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
    for individual, fitness in zip(invalid_individuals, fitnesses):
        individual.fitness.values = fitness


def _apply_crossover(
    toolbox: Toolbox,
    offspring: MutableSequence[Individual],
    cx_pb: float,
    rng: np.random.Generator,
) -> None:
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


def _apply_mutation(
    toolbox: Toolbox,
    offspring: MutableSequence[Individual],
    mut_pb: float,
    rng: np.random.Generator,
) -> None:
    for i in range(len(offspring)):
        if rng.random() < mut_pb:
            (offspring[i],) = toolbox.mutate(offspring[i])
            try:
                del offspring[i].fitness.values
            except AttributeError:
                pass


def _record_generation_metrics(
    generation: int,
    population: Population,
    hall_of_fame: HallOfFame | None,
    statistics: Statistics | None,
    logbook: Logbook | None,
    history_logger: HistoryLogger | None,
    checkpoint_callback: Callable[[int, TrainingResult], None] | None,
) -> None:
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
