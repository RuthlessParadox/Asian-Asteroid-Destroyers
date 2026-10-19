from collections.abc import Sequence
from typing import Callable, cast

from deap import tools

from asian_asteroid_destroyers.genetic_algorithm.components import (
    Individual,
)


def crossover_one_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOnePoint,
    )
    return typed_cx_one_point(individual_1, individual_2)


def crossover_two_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxTwoPoint,
    )
    return typed_cx_two_point(individual_1, individual_2)


def crossover_uniform(
    individual_1: Individual,
    individual_2: Individual,
    independent_probability: float,
) -> tuple[Individual, Individual]:
    typed_cx_uniform = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniform,
    )
    return typed_cx_uniform(
        individual_1,
        individual_2,
        independent_probability,
    )


def crossover_partially_matched(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_partially_matched = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxPartialyMatched,
    )
    return typed_cx_partially_matched(individual_1, individual_2)


def crossover_uniform_partially_matched(
    individual_1: Individual,
    individual_2: Individual,
    independent_probability: float,
) -> tuple[Individual, Individual]:
    typed_cx_uniform_partially_matched = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniformPartialyMatched,
    )
    return typed_cx_uniform_partially_matched(
        individual_1,
        individual_2,
        independent_probability,
    )


def crossover_ordered(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_ordered = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOrdered,
    )
    return typed_cx_ordered(individual_1, individual_2)


def crossover_blend(
    individual_1: Individual,
    individual_2: Individual,
    alpha: float,
) -> tuple[Individual, Individual]:
    typed_cx_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxBlend,
    )
    return typed_cx_blend(individual_1, individual_2, alpha)


def crossover_evolution_strategy_blend(
    individual_1: Individual,
    individual_2: Individual,
    alpha: float,
) -> tuple[Individual, Individual]:
    typed_cx_evolution_strategy_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxESBlend,
    )
    return typed_cx_evolution_strategy_blend(individual_1, individual_2, alpha)


def crossover_evolution_strategy_two_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_evolution_strategy_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxESTwoPoint,
    )
    return typed_cx_evolution_strategy_two_point(individual_1, individual_2)


def crossover_simulated_binary(
    individual_1: Individual,
    individual_2: Individual,
    eta: float,
) -> tuple[Individual, Individual]:
    typed_cx_simulated_binary = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxSimulatedBinary,
    )
    return typed_cx_simulated_binary(individual_1, individual_2, eta)


def crossover_simulated_binary_bounded(
    individual_1: Individual,
    individual_2: Individual,
    eta: float,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
) -> tuple[Individual, Individual]:
    typed_cx_simulated_binary_bounded = cast(
        Callable[
            [
                Individual,
                Individual,
                float,
                float | Sequence[float],
                float | Sequence[float],
            ],
            tuple[Individual, Individual],
        ],
        tools.cxSimulatedBinaryBounded,
    )
    return typed_cx_simulated_binary_bounded(
        individual_1, individual_2, eta, lower_bound, upper_bound
    )


def crossover_messy_one_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_messy_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxMessyOnePoint,
    )
    return typed_cx_messy_one_point(individual_1, individual_2)
