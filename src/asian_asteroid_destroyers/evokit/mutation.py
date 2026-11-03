from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from deap import tools

from .individual import Individual

if TYPE_CHECKING:
    from asian_asteroid_destroyers.chromosomes import ControllerChromosome

    from .types import MutationOperator


def mut_fuzzy_gaussian(
    template: ControllerChromosome,
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    independent_probability: float = 0.2,
) -> MutationOperator:
    """Closure factory providing a highly optimized vectorized mutation operator.

    Args:
        template: The structural chromosome configuration used to extract the
            exact flat geometric indices.
        rng: The explicit dependency-injected random number generator.
        mu: Mean of the Gaussian noise distribution.
        sigma: Standard deviation of the Gaussian step.
        independent_probability: Independent probability for each specific gene to mutate.
    """

    membership_function_slices: list[tuple[int, int]] = []
    current_index = 0

    for blueprint in (
        template.thrust_blueprint,
        template.turn_blueprint,
        template.fire_blueprint,
    ):
        for param in blueprint.membership_function_pool:
            param_length = len(param)
            membership_function_slices.append(
                (current_index, current_index + param_length),
            )
            current_index += param_length

        for rule in blueprint.rules:
            current_index += len(rule.consequent_coefficients) + 1

    def mut_fuzzy_gaussian_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        """Applies vectorized Gaussian noise and repairs bounds in place."""
        genes = np.array(individual, dtype=np.float64)

        mutation_mask = rng.random(size=genes.size) < independent_probability

        if np.any(mutation_mask):
            noise = rng.normal(mu, sigma, size=genes.size)
            genes += noise * mutation_mask

            for start, end in membership_function_slices:
                genes[start:end].sort()

            individual[:] = genes.tolist()

        return (individual,)

    return mut_fuzzy_gaussian_wrapper


def mut_gaussian(
    mu: float | Sequence[float],
    sigma: float | Sequence[float],
    independent_probability: float,
) -> MutationOperator:
    typed_mut_gaussian = cast(
        Callable[
            [
                Individual,
                float | Sequence[float],
                float | Sequence[float],
                float,
            ],
            tuple[Individual],
        ],
        tools.mutGaussian,
    )

    def mut_gaussian_wrapper(individual: Individual) -> tuple[Individual]:
        return typed_mut_gaussian(
            individual,
            mu,
            sigma,
            independent_probability,
        )

    return mut_gaussian_wrapper


def mut_shuffle_indices(
    independent_probability: float,
) -> MutationOperator:
    typed_mut_shuffle_indices = cast(
        Callable[[Individual, float], tuple[Individual]],
        tools.mutShuffleIndexes,
    )

    def mut_shuffle_indices_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_shuffle_indices(individual, independent_probability)

    return mut_shuffle_indices_wrapper


def mut_flip_bit(independent_probability: float) -> MutationOperator:
    typed_mut_flip_bit = cast(
        Callable[[Individual, float], tuple[Individual]],
        tools.mutFlipBit,
    )

    def mut_flip_bit_wrapper(individual: Individual) -> tuple[Individual]:
        return typed_mut_flip_bit(individual, independent_probability)

    return mut_flip_bit_wrapper


def mut_uniform_int(
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    independent_probability: float,
) -> MutationOperator:
    typed_mut_uniform_int = cast(
        Callable[
            [
                Individual,
                float | Sequence[float],
                float | Sequence[float],
                float,
            ],
            tuple[Individual],
        ],
        tools.mutUniformInt,
    )

    def mut_uniform_int_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_uniform_int(
            individual,
            lower_bound,
            upper_bound,
            independent_probability,
        )

    return mut_uniform_int_wrapper


def mut_polynomial_bounded(
    eta: float,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    independent_probability: float,
) -> MutationOperator:
    typed_mut_polynomial_bounded = cast(
        Callable[
            [
                Individual,
                float,
                float | Sequence[float],
                float | Sequence[float],
                float,
            ],
            tuple[Individual],
        ],
        tools.mutPolynomialBounded,
    )

    def mut_polynomial_bounded_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_polynomial_bounded(
            individual,
            eta,
            lower_bound,
            upper_bound,
            independent_probability,
        )

    return mut_polynomial_bounded_wrapper


def mut_evolution_strategy_log_normal(
    c: float,
    independent_probability: float,
) -> MutationOperator:
    typed_mut_es_log_normal = cast(
        Callable[[Individual, float, float], tuple[Individual]],
        tools.mutESLogNormal,
    )

    def mut_evolution_strategy_log_normal_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_es_log_normal(individual, c, independent_probability)

    return mut_evolution_strategy_log_normal_wrapper
