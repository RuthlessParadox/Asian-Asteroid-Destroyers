from collections.abc import Sequence
from typing import Callable, cast

from deap import tools

from .individual import Individual
from .types import CrossoverOperator


def cx_one_point() -> CrossoverOperator:
    typed_cx_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOnePoint,
    )

    def cx_one_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_one_point(individual_1, individual_2)

    return cx_one_point_wrapper


def cx_two_point() -> CrossoverOperator:
    typed_cx_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxTwoPoint,
    )

    def cx_two_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_two_point(individual_1, individual_2)

    return cx_two_point_wrapper


def cx_uniform(independent_probability: float) -> CrossoverOperator:
    typed_cx_uniform = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniform,
    )

    def cx_uniform_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_uniform(
            individual_1,
            individual_2,
            independent_probability,
        )

    return cx_uniform_wrapper


def cx_partially_matched() -> CrossoverOperator:
    typed_cx_partially_matched = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxPartialyMatched,
    )

    def cx_partially_matched_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_partially_matched(individual_1, individual_2)

    return cx_partially_matched_wrapper


def cx_uniform_partially_matched(
    independent_probability: float,
) -> CrossoverOperator:
    typed_cx_uniform_partially_matched = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniformPartialyMatched,
    )

    def cx_uniform_partially_matched_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_uniform_partially_matched(
            individual_1,
            individual_2,
            independent_probability,
        )

    return cx_uniform_partially_matched_wrapper


def cx_ordered() -> CrossoverOperator:
    typed_cx_ordered = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOrdered,
    )

    def cx_ordered_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_ordered(individual_1, individual_2)

    return cx_ordered_wrapper


def cx_blend(alpha: float) -> CrossoverOperator:
    typed_cx_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxBlend,
    )

    def cx_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_blend(individual_1, individual_2, alpha)

    return cx_blend_wrapper


def cx_evolution_strategy_blend(alpha: float) -> CrossoverOperator:
    typed_cx_es_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxESBlend,
    )

    def cx_evolution_strategy_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_es_blend(individual_1, individual_2, alpha)

    return cx_evolution_strategy_blend_wrapper


def cx_evolution_strategy_two_point() -> CrossoverOperator:
    typed_cx_es_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxESTwoPoint,
    )

    def cx_evolution_strategy_two_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_es_two_point(individual_1, individual_2)

    return cx_evolution_strategy_two_point_wrapper


def cx_simulated_binary(eta: float) -> CrossoverOperator:
    typed_cx_simulated_binary = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxSimulatedBinary,
    )

    def cx_simulated_binary_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_simulated_binary(individual_1, individual_2, eta)

    return cx_simulated_binary_wrapper


def cx_simulated_binary_bounded(
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

    def cx_simulated_binary_bounded_wrapper(
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

    return cx_simulated_binary_bounded_wrapper


def cx_messy_one_point() -> CrossoverOperator:
    typed_cx_messy_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxMessyOnePoint,
    )

    def cx_messy_one_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_messy_one_point(individual_1, individual_2)

    return cx_messy_one_point_wrapper
