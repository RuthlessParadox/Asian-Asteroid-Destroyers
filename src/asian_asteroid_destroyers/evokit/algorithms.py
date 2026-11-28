from __future__ import annotations

import signal
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar

import numpy as np
from pathos.multiprocessing import ProcessPool

from .seeding import derive_generation_seed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, MutableSequence

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


R_co = TypeVar("R_co", bound=TrainingResult, covariant=True)


class EvolutionAlgorithm(Protocol[R_co]):
    __name__: str
    __module__: str

    def __call__(
        self,
        toolbox: Toolbox,
        cx_pb: float,
        mut_pb: float,
        generations: int,
        population_size: int,
        *,
        seed: int | np.random.Generator | None = None,
        master_seed: int | None = None,
        workers: int | None = None,
        hall_of_fame: HallOfFame | None = None,
        statistics: Statistics | None = None,
        logbook: Logbook | None = None,
        history_logger: HistoryLogger | None = None,
        checkpoint_callback: Callable[[int, R_co], None] | None = None,
        initial_population: Population | None = None,
        start_generation: int = 1,
    ) -> R_co: ...


def run_nsga_evolution(
    toolbox: Toolbox,
    cx_pb: float,
    mut_pb: float,
    generations: int,
    population_size: int,
    *,
    seed: int | np.random.Generator | None = None,
    master_seed: int | None = None,
    workers: int | None = None,
    hall_of_fame: HallOfFame | None = None,
    statistics: Statistics | None = None,
    logbook: Logbook | None = None,
    history_logger: HistoryLogger | None = None,
    checkpoint_callback: Callable[[int, TrainingResult], None] | None = None,
    initial_population: Population | None = None,
    start_generation: int = 1,
) -> TrainingResult:
    _validate_parameters(
        cx_pb,
        mut_pb,
        generations,
        population_size,
        initial_population,
        start_generation,
    )

    resolved_master_seed = (
        master_seed
        if master_seed is not None
        else int(np.random.default_rng().integers(1, 2**31 - 1))
    )

    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )
    population = (
        initial_population
        if initial_population is not None
        else toolbox.population(population_size)
    )

    with _create_process_pool(workers) as executor:
        if start_generation == 1:
            _evaluate_seeded_individuals(
                pool=executor,
                evaluate_function=toolbox.evaluate,
                individuals=population,
                generation_seed=derive_generation_seed(
                    resolved_master_seed, 0
                ),
            )
            _record_generation_metrics(
                generation=0,
                population=population,
                hall_of_fame=hall_of_fame,
                statistics=statistics,
                logbook=logbook,
                history_logger=history_logger,
                checkpoint_callback=checkpoint_callback,
            )

        for generation in range(start_generation, generations + 1):
            parents = toolbox.select_parents(population, population_size)
            offspring = [toolbox.clone(individual) for individual in parents]

            _apply_crossover(toolbox, offspring, cx_pb, rng)
            _apply_mutation(toolbox, offspring, mut_pb, rng)

            _evaluate_seeded_individuals(
                pool=executor,
                evaluate_function=toolbox.evaluate,
                individuals=offspring,
                generation_seed=derive_generation_seed(
                    resolved_master_seed, generation
                ),
            )

            population = toolbox.select_survivors(
                [*population, *offspring],
                population_size,
            )

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
    initial_population: Population | None,
    start_generation: int,
) -> None:
    if not (0.0 <= cx_pb <= 1.0):
        raise ValueError(
            "Crossover probability 'cx_pb' must be between 0.0 and 1.0. "
            + f"Got: {cx_pb}"
        )
    if not (0.0 <= mut_pb <= 1.0):
        raise ValueError(
            "Mutation probability 'mut_pb' must be between 0.0 and 1.0. "
            + f"Got: {mut_pb}"
        )
    if generations < 0:
        raise ValueError(
            "Number of generations 'n_gen' must be non-negative. "
            + f"Got: {generations}"
        )
    if population_size <= 0:
        raise ValueError(
            "Population size 'population_size' must be a positive integer. "
            + f"Got: {population_size}"
        )
    if start_generation < 1:
        raise ValueError(
            "Start generation 'start_generation' must be a positive integer. "
            + f"Got: {start_generation}"
        )
    if start_generation > generations + 1:
        raise ValueError(
            f"Start generation 'start_generation' ({start_generation}) "
            + f"cannot exceed 'generations' + 1 ({generations + 1})."
        )
    if start_generation > 1 and initial_population is None:
        raise ValueError(
            "Initial population 'initial_population' cannot be None with "
            + f"mid-point generation {start_generation}."
        )
    if (
        initial_population is not None
        and len(initial_population) != population_size
    ):
        raise ValueError(
            f"The length of 'initial_population' ({len(initial_population)}) "
            + f"does not match 'population_size' ({population_size})."
        )


def _create_process_pool(workers: int | None) -> ProcessPool:
    if workers is not None:
        if workers <= 0:
            raise ValueError(
                f"'workers' must be a positive integer or None. Got: {workers}"
            )
        return ProcessPool(initializer=_ignore_sigint, nodes=workers)
    return ProcessPool(initializer=_ignore_sigint)


def _ignore_sigint() -> None:
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)


def _evaluate_seeded_individuals(
    pool: ProcessPool,
    evaluate_function: Callable[[Individual, int], tuple[float, ...]],
    individuals: Population,
    generation_seed: int,
) -> None:
    invalid = [
        individual
        for individual in individuals
        if not individual.fitness.valid
    ]
    if not invalid:
        return

    seeds = [generation_seed] * len(invalid)
    typed_map: Callable[
        [
            Callable[[Individual, int], tuple[float, ...]],
            list[Individual],
            list[int],
        ],
        Iterable[tuple[float, ...]],
    ] = pool.map  # pyright: ignore[reportAssignmentType]
    fitnesses = list(typed_map(evaluate_function, invalid, seeds))
    for individual, fitness in zip(invalid, fitnesses):
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
            del offspring[i - 1].fitness.values
            del offspring[i].fitness.values


def _apply_mutation(
    toolbox: Toolbox,
    offspring: MutableSequence[Individual],
    mut_pb: float,
    rng: np.random.Generator,
) -> None:
    for i in range(len(offspring)):
        if rng.random() < mut_pb:
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values


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
