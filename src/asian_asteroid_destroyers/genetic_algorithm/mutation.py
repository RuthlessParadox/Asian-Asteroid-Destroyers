from collections.abc import Sequence
from typing import Callable, cast

from deap import tools

from .individual import Individual
from .types import MutationOperator


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
