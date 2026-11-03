from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from deap import tools

from asian_asteroid_destroyers.chromosomes import (
    flatten_chromosome,
    hydrate_chromosome_blueprint,
    repair_geometry_invariants,
)

from .individual import Individual

if TYPE_CHECKING:
    from asian_asteroid_destroyers.chromosomes import ControllerChromosome

    from .types import CrossoverOperator


def cx_fuzzy_blend(
    template: ControllerChromosome,
    rng: np.random.Generator,
    alpha: float = 0.3,
) -> CrossoverOperator:
    """Closure factory providing a dependency-injected blend crossover.

    Args:
        template: The structural chromosome design used as a hydration map.
        rng: The explicit random number generator instance injected from the
            outer evolutionary training environment.
        alpha: Blend exploration factor. Controls how far outside the parent
            bounds the offspring genes can blend.
    """

    def cx_fuzzy_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        """Applies aligned segment swaps and numeric BLX blending via rng."""

        parent_1 = hydrate_chromosome_blueprint(individual_1[:], template)
        parent_2 = hydrate_chromosome_blueprint(individual_2[:], template)

        subsystems = [
            (parent_1.thrust_blueprint, parent_2.thrust_blueprint),
            (parent_1.turn_blueprint, parent_2.turn_blueprint),
            (parent_1.fire_blueprint, parent_2.fire_blueprint),
        ]

        for subsystem_1, subsystem_2 in subsystems:
            # Segment-wise crossover: Membership Function pools
            pool_1 = subsystem_1.membership_function_pool
            pool_2 = subsystem_2.membership_function_pool

            for index in range(len(pool_1)):
                membership_function_1 = pool_1[index]
                membership_function_2 = pool_2[index]

                if rng.random() < 0.7:
                    parent_1_array = np.array(
                        membership_function_1,
                        dtype=np.float64,
                    )
                    parent_2_array = np.array(
                        membership_function_2,
                        dtype=np.float64,
                    )

                    min_values = np.minimum(parent_1_array, parent_2_array)
                    max_values = np.maximum(parent_1_array, parent_2_array)
                    differences = max_values - min_values

                    low_bounds = min_values - alpha * differences
                    high_bounds = max_values + alpha * differences

                    offspring_1 = rng.uniform(low_bounds, high_bounds)
                    offspring_2 = rng.uniform(low_bounds, high_bounds)

                    pool_1[index] = type(membership_function_1)(*offspring_1)
                    pool_2[index] = type(membership_function_2)(*offspring_2)
                else:
                    pool_1[index], pool_2[index] = pool_2[index], pool_1[index]

            # Segment-wise crossover: Rule Consequents
            rules_1 = subsystem_1.rules
            rules_2 = subsystem_2.rules
            for index in range(len(rules_1)):
                if rng.random() < 0.5:
                    rules_1[index], rules_2[index] = (
                        rules_2[index],
                        rules_1[index],
                    )

        individual_1[:] = repair_geometry_invariants(
            flatten_chromosome(parent_1),
            template,
        )
        individual_2[:] = repair_geometry_invariants(
            flatten_chromosome(parent_2),
            template,
        )

        return individual_1, individual_2

    return cx_fuzzy_blend_wrapper


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
