from collections.abc import Sequence
from typing import Callable, cast

from deap import tools

from .individual import Individual
from .types import CrossoverOperator


def crossover_one_point() -> CrossoverOperator:
    typed_cx_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOnePoint,
    )

    def crossover_one_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_one_point(individual_1, individual_2)

    return crossover_one_point_wrapper


def crossover_two_point() -> CrossoverOperator:
    typed_cx_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxTwoPoint,
    )

    def crossover_two_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_two_point(individual_1, individual_2)

    return crossover_two_point_wrapper


def crossover_uniform(independent_probability: float) -> CrossoverOperator:
    typed_cx_uniform = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniform,
    )

    def crossover_uniform_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_uniform(
            individual_1,
            individual_2,
            independent_probability,
        )

    return crossover_uniform_wrapper


def crossover_partially_matched() -> CrossoverOperator:
    typed_cx_partially_matched = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxPartialyMatched,
    )

    def crossover_partially_matched_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_partially_matched(individual_1, individual_2)

    return crossover_partially_matched_wrapper


def crossover_uniform_partially_matched(
    independent_probability: float,
) -> CrossoverOperator:
    typed_cx_uniform_partially_matched = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniformPartialyMatched,
    )

    def crossover_uniform_partially_matched_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_uniform_partially_matched(
            individual_1,
            individual_2,
            independent_probability,
        )

    return crossover_uniform_partially_matched_wrapper


def crossover_ordered() -> CrossoverOperator:
    typed_cx_ordered = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOrdered,
    )

    def crossover_ordered_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_ordered(individual_1, individual_2)

    return crossover_ordered_wrapper


def crossover_blend(alpha: float) -> CrossoverOperator:
    typed_cx_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxBlend,
    )

    def crossover_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_blend(individual_1, individual_2, alpha)

    return crossover_blend_wrapper


def crossover_evolution_strategy_blend(alpha: float) -> CrossoverOperator:
    typed_cx_es_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxESBlend,
    )

    def crossover_evolution_strategy_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_es_blend(individual_1, individual_2, alpha)

    return crossover_evolution_strategy_blend_wrapper


def crossover_evolution_strategy_two_point() -> CrossoverOperator:
    typed_cx_es_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxESTwoPoint,
    )

    def crossover_evolution_strategy_two_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_es_two_point(individual_1, individual_2)

    return crossover_evolution_strategy_two_point_wrapper


def crossover_simulated_binary(eta: float) -> CrossoverOperator:
    typed_cx_simulated_binary = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxSimulatedBinary,
    )

    def crossover_simulated_binary_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_simulated_binary(individual_1, individual_2, eta)

    return crossover_simulated_binary_wrapper


def crossover_simulated_binary_bounded(
    eta: float,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
) -> CrossoverOperator:
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

    def crossover_simulated_binary_bounded_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_simulated_binary_bounded(
            individual_1,
            individual_2,
            eta,
            lower_bound,
            upper_bound,
        )

    return crossover_simulated_binary_bounded_wrapper


def crossover_messy_one_point() -> CrossoverOperator:
    typed_cx_messy_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxMessyOnePoint,
    )

    def crossover_messy_one_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_messy_one_point(individual_1, individual_2)

    return crossover_messy_one_point_wrapper
