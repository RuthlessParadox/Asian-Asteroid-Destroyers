from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from deap import tools

from asian_asteroid_destroyers.chromosomes import (
    ControllerChromosome,
    MFParamTrapezoidal,
    MFParamTriangular,
    RuleGene,
    SubsystemChromosome,
    flatten_chromosome,
    hydrate_chromosome_blueprint,
    repair_geometry_invariants,
)

from .individual import Individual

if TYPE_CHECKING:
    from .types import CrossoverOperator


def cx_fuzzy_blend(
    template: ControllerChromosome,
    rng: np.random.Generator,
    alpha: float = 0.3,
) -> CrossoverOperator:
    def cx_fuzzy_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        parent_1 = hydrate_chromosome_blueprint(individual_1[:], template)
        parent_2 = hydrate_chromosome_blueprint(individual_2[:], template)

        child_1_blueprints: list[SubsystemChromosome] = []
        child_2_blueprints: list[SubsystemChromosome] = []

        subsystems = [
            (parent_1.thrust_blueprint, parent_2.thrust_blueprint),
            (parent_1.turn_blueprint, parent_2.turn_blueprint),
            (parent_1.fire_blueprint, parent_2.fire_blueprint),
        ]

        for subsystem_1, subsystem_2 in subsystems:
            child_1_pool: list[MFParamTriangular | MFParamTrapezoidal] = []
            child_2_pool: list[MFParamTriangular | MFParamTrapezoidal] = []

            for membership_function_1, membership_function_2 in zip(
                subsystem_1.membership_function_pool,
                subsystem_2.membership_function_pool,
            ):
                if rng.random() < 0.7:
                    array_1 = np.array(membership_function_1, dtype=np.float64)
                    array_2 = np.array(membership_function_2, dtype=np.float64)

                    min_values = np.minimum(array_1, array_2)
                    max_values = np.maximum(array_1, array_2)
                    differences = max_values - min_values

                    low_bounds = min_values - alpha * differences
                    high_bounds = max_values + alpha * differences

                    offspring_1 = rng.uniform(low_bounds, high_bounds)
                    offspring_2 = rng.uniform(low_bounds, high_bounds)

                    if len(membership_function_1) == 3:
                        child_1_pool.append(
                            MFParamTriangular(
                                cast(float, offspring_1[0]),
                                cast(float, offspring_1[1]),
                                cast(float, offspring_1[2]),
                            )
                        )
                        child_2_pool.append(
                            MFParamTriangular(
                                cast(float, offspring_2[0]),
                                cast(float, offspring_2[1]),
                                cast(float, offspring_2[2]),
                            )
                        )
                    else:
                        child_1_pool.append(
                            MFParamTrapezoidal(
                                cast(float, offspring_1[0]),
                                cast(float, offspring_1[1]),
                                cast(float, offspring_1[2]),
                                cast(float, offspring_1[3]),
                            )
                        )
                        child_2_pool.append(
                            MFParamTrapezoidal(
                                cast(float, offspring_2[0]),
                                cast(float, offspring_2[1]),
                                cast(float, offspring_2[2]),
                                cast(float, offspring_2[3]),
                            )
                        )
                else:
                    child_1_pool.append(membership_function_2)
                    child_2_pool.append(membership_function_1)

            child_1_rules: list[RuleGene] = []
            child_2_rules: list[RuleGene] = []

            for rule_1, rule_2 in zip(subsystem_1.rules, subsystem_2.rules):
                if rng.random() < 0.5:
                    child_1_rules.append(
                        RuleGene(
                            antecedents=rule_2.antecedents,
                            consequent_coefficients=(
                                rule_2.consequent_coefficients
                            ),
                            consequent_offset=rule_2.consequent_offset,
                        )
                    )
                    child_2_rules.append(
                        RuleGene(
                            antecedents=rule_1.antecedents,
                            consequent_coefficients=(
                                rule_1.consequent_coefficients
                            ),
                            consequent_offset=rule_1.consequent_offset,
                        )
                    )
                else:
                    child_1_rules.append(
                        RuleGene(
                            antecedents=rule_1.antecedents,
                            consequent_coefficients=(
                                rule_1.consequent_coefficients
                            ),
                            consequent_offset=rule_1.consequent_offset,
                        )
                    )
                    child_2_rules.append(
                        RuleGene(
                            antecedents=rule_2.antecedents,
                            consequent_coefficients=(
                                rule_2.consequent_coefficients
                            ),
                            consequent_offset=rule_2.consequent_offset,
                        )
                    )

            child_1_blueprints.append(
                SubsystemChromosome(
                    membership_function_pool=child_1_pool,
                    rules=child_1_rules,
                )
            )
            child_2_blueprints.append(
                SubsystemChromosome(
                    membership_function_pool=child_2_pool,
                    rules=child_2_rules,
                )
            )

        final_controller_1 = ControllerChromosome(
            thrust_blueprint=child_1_blueprints[0],
            turn_blueprint=child_1_blueprints[1],
            fire_blueprint=child_1_blueprints[2],
        )
        final_controller_2 = ControllerChromosome(
            thrust_blueprint=child_2_blueprints[0],
            turn_blueprint=child_2_blueprints[1],
            fire_blueprint=child_2_blueprints[2],
        )

        individual_1[:] = repair_geometry_invariants(
            flatten_chromosome(final_controller_1),
            template,
        )
        individual_2[:] = repair_geometry_invariants(
            flatten_chromosome(final_controller_2),
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
