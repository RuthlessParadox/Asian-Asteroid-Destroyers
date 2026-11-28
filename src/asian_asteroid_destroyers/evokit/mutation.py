from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from deap import tools

from asian_asteroid_destroyers.chromosomes import (
    create_geometry_repair_maps,
    repair_geometry_invariants,
)

from .individual import Individual

if TYPE_CHECKING:
    from asian_asteroid_destroyers.chromosomes import (
        ControllerChromosome,
        FlatGenome,
    )

    from .types import MutationOperator


def mut_fuzzy_gaussian(
    template: ControllerChromosome,
    rng: np.random.Generator,
    mu: float = 0.0,
    sigma: float = 0.05,
    independent_probability: float = 0.05,
    gate_flip_probability: float = 0.1,
) -> MutationOperator:
    triangular_array, trapezoidal_array = create_geometry_repair_maps(template)

    gate_indices: list[int] = []
    current_index = 0
    for blueprint in (
        template.thrust_offensive,
        template.thrust_defensive,
        template.turn_offensive,
        template.turn_defensive,
        template.threat_gate,
    ):
        for param in blueprint.membership_function_pool:
            current_index += len(param)
        for rule in blueprint.rules:
            current_index += len(rule.consequent_coefficients) + 1
            gate_indices.append(current_index)
            current_index += 1

    def mut_fuzzy_gaussian_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        genes = np.array(individual, dtype=np.float64)
        param_mask = rng.random(size=genes.size) < independent_probability
        for gate_index in gate_indices:
            param_mask[gate_index] = False

        if np.any(param_mask):
            noise = rng.normal(mu, sigma, size=genes.size)
            genes[param_mask] += noise[param_mask]

        for gate_index in gate_indices:
            if rng.random() < gate_flip_probability:
                genes[gate_index] = 1.0 - genes[gate_index]

        genes_flat: FlatGenome = genes.tolist()  # pyright: ignore[reportAny]

        individual[:] = repair_geometry_invariants(
            genes_flat,
            triangular_array,
            trapezoidal_array,
        )

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
