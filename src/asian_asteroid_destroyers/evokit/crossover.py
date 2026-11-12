from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from deap import tools

from asian_asteroid_destroyers.chromosomes import (
    create_geometry_repair_maps,
    repair_geometry_invariants,
)

from .individual import Individual

if TYPE_CHECKING:
    import numpy.typing as npt

    from asian_asteroid_destroyers.chromosomes import (
        ControllerChromosome,
        FlatGenome,
    )

    from .types import CrossoverOperator


def cx_fuzzy_blend(
    template: ControllerChromosome,
    rng: np.random.Generator,
    alpha: float = 0.25,
) -> CrossoverOperator:
    triangular_array, trapezoidal_array = create_geometry_repair_maps(template)
    membership_function_blocks: list[npt.NDArray[np.intp]] = []
    rule_blocks: list[npt.NDArray[np.intp]] = []

    subsystems = (
        template.thrust_offensive,
        template.thrust_defensive,
        template.turn_offensive,
        template.turn_defensive,
        template.threat_gate,
    )

    current_index = 0
    for blueprint in subsystems:
        membership_function_pool_start = current_index
        for param in blueprint.membership_function_pool:
            current_index += len(param)
        membership_function_blocks.append(
            np.arange(
                membership_function_pool_start,
                current_index,
                dtype=np.intp,
            )
        )

        for rule in blueprint.rules:
            rule_start = current_index
            current_index += len(rule.consequent_coefficients) + 2
            rule_blocks.append(
                np.arange(rule_start, current_index, dtype=np.intp)
            )

    def cx_fuzzy_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        array_1 = np.array(individual_1, dtype=np.float64)
        array_2 = np.array(individual_2, dtype=np.float64)

        output_1 = array_1.copy()
        output_2 = array_2.copy()

        membership_function_random = cast(
            Iterable[float],
            rng.random(size=len(membership_function_blocks)),
        )
        membership_function_blend_indices: list[int] = []
        membership_function_swap_indices: list[int] = []

        for index, random_value in enumerate(membership_function_random):
            if random_value < 0.7:
                membership_function_blend_indices.extend(
                    membership_function_blocks[index]
                )
            else:
                membership_function_swap_indices.extend(
                    membership_function_blocks[index]
                )

        if membership_function_blend_indices:
            blend_indices = np.array(
                membership_function_blend_indices,
                dtype=np.intp,
            )
            min_values = np.minimum(
                array_1[blend_indices], array_2[blend_indices]
            )
            max_values = np.maximum(
                array_1[blend_indices], array_2[blend_indices]
            )
            differences = max_values - min_values

            lows = min_values - alpha * differences
            highs = max_values + alpha * differences

            blended = rng.uniform(lows, highs)
            output_1[blend_indices] = blended
            output_2[blend_indices] = blended

        if membership_function_swap_indices:
            swap_indices = np.array(
                membership_function_swap_indices,
                dtype=np.intp,
            )
            output_1[swap_indices] = array_2[swap_indices]
            output_2[swap_indices] = array_1[swap_indices]

        rule_random = cast(Iterable[float], rng.random(size=len(rule_blocks)))
        rule_swap_indices: list[int] = []

        for index, random_value in enumerate(rule_random):
            if random_value < 0.5:
                rule_swap_indices.extend(rule_blocks[index])

        if rule_swap_indices:
            rule_indices = np.array(rule_swap_indices, dtype=np.intp)
            output_1[rule_indices] = array_2[rule_indices]
            output_2[rule_indices] = array_1[rule_indices]

        output_1_flat: FlatGenome = output_1.tolist()  # pyright: ignore[reportAny]
        output_2_flat: FlatGenome = output_2.tolist()  # pyright: ignore[reportAny]

        individual_1[:] = repair_geometry_invariants(
            output_1_flat,
            triangular_array,
            trapezoidal_array,
        )
        individual_2[:] = repair_geometry_invariants(
            output_2_flat,
            triangular_array,
            trapezoidal_array,
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
