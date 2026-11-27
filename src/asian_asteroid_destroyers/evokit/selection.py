from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from deap import tools

from .types import Population

if TYPE_CHECKING:
    from .types import (
        BoolMatrix,
        FloatMatrix,
        FloatVector,
        IntMatrix,
        IntVector,
        SelectionOperator,
    )


def sel_tournament(
    tournament_size: int,
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_tournament = cast(
        Callable[[Population, int, int, str], Population],
        tools.selTournament,
    )

    def sel_tournament_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_tournament(
            population,
            k,
            tournament_size,
            fitness_attribute,
        )

    return sel_tournament_wrapper


def sel_roulette(
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_roulette = cast(
        Callable[[Population, int, str], Population],
        tools.selRoulette,
    )

    def sel_roulette_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_roulette(population, k, fitness_attribute)

    return sel_roulette_wrapper


def sel_nsga_2(
    algorithm: str = "standard",
) -> SelectionOperator:
    typed_sel_nsga_2 = cast(
        Callable[[Population, int, str], Population],
        tools.selNSGA2,
    )

    def sel_nsga_2_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_nsga_2(population, k, algorithm)

    return sel_nsga_2_wrapper


_DOMINANCE_CHUNK_SIZE = 64
_ASF_PENALTY_SCALE = 1e6
_NORM_EPSILON = 1e-18
_INTERCEPT_EPSILON = 1e-12
_TIE_EPSILON = 1e-15


def sel_nsga_3(
    objective_count: int = 5,
    subdivisions: int = 4,
    *,
    inner_subdivisions: int | None = None,
    inner_scale: float = 0.5,
    seed: int | np.random.Generator | None = None,
) -> SelectionOperator:
    if objective_count < 2:
        raise ValueError(
            f"'objective_count' must be at least 2. Got: {objective_count}"
        )
    if subdivisions < 1:
        raise ValueError(
            f"'subdivisions' must be a positive integer. Got: {subdivisions}"
        )
    if inner_subdivisions is not None and inner_subdivisions < 1:
        raise ValueError(
            "'inner_subdivisions' must be a positive integer or None. "
            + f"Got: {inner_subdivisions}"
        )
    if not (0.0 < inner_scale < 1.0):
        raise ValueError(
            f"'inner_scale' must be between 0.0 and 1.0. Got: {inner_scale}"
        )

    reference_points = _build_reference_points(
        objective_count=objective_count,
        subdivisions=subdivisions,
        inner_subdivisions=inner_subdivisions,
        inner_scale=inner_scale,
    )
    unit_reference_points = _normalize_reference_directions(reference_points)
    other_axis_indices = _build_other_axis_indices(objective_count)
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )
    cached_weights: tuple[float, ...] | None = None
    cached_min_sign: FloatVector | None = None

    def sel_nsga_3_wrapper(population: Population, k: int) -> Population:
        nonlocal cached_weights, cached_min_sign

        population_size = len(population)
        if population_size == 0:
            raise ValueError("'population' must not be empty.")
        if k <= 0 or k > population_size:
            raise ValueError(
                f"'k' must be in [1, {population_size}]. Got: {k}"
            )

        objectives, cached_weights, cached_min_sign = (
            _extract_minimization_objectives(
                population=population,
                expected_objectives=objective_count,
                cached_weights=cached_weights,
                cached_min_sign=cached_min_sign,
            )
        )
        fronts = _non_dominated_sort(objectives, max_selected=k)

        selected_indices: list[int] = []
        front_index = 0
        front_count = len(fronts)
        while front_index < front_count:
            current_front = fronts[front_index]
            if len(selected_indices) + len(current_front) <= k:
                selected_indices.extend(current_front)
                front_index += 1
            else:
                break

        remaining_to_select_count = k - len(selected_indices)
        if remaining_to_select_count == 0:
            return [population[index] for index in selected_indices]

        critical_front = fronts[front_index]
        combined_indices = selected_indices + critical_front
        # noinspection PyTypeChecker
        normalized = _normalize_objectives(
            objectives=(  # pyright: ignore[reportArgumentType]
                objectives[combined_indices]
            ),
            objective_count=objective_count,
            other_axis_indices=other_axis_indices,
        )
        chosen_from_critical = _niching_selection(
            normalized_objectives=normalized,
            selected_count=len(selected_indices),
            critical_front=critical_front,
            unit_reference_points=unit_reference_points,
            remaining_to_select_count=remaining_to_select_count,
            rng=rng,
        )
        selected_indices.extend(chosen_from_critical)
        return [population[index] for index in selected_indices]

    return sel_nsga_3_wrapper


def _build_other_axis_indices(objective_count: int) -> IntMatrix:
    rows: list[list[int]] = [
        [column for column in range(objective_count) if column != axis]
        for axis in range(objective_count)
    ]
    # noinspection PyTypeChecker
    return (  # pyright: ignore[reportReturnType]
        np.asarray(rows, dtype=np.int64)
    )


def _generate_reference_points(
    objective_count: int,
    subdivisions: int,
) -> FloatMatrix:
    divider_combinations = itertools.combinations(
        range(subdivisions + objective_count - 1),
        objective_count - 1,
    )
    dividers = np.fromiter(
        itertools.chain.from_iterable(divider_combinations),
        dtype=np.int64,
    ).reshape(-1, objective_count - 1)

    padded = np.empty((dividers.shape[0], objective_count + 1), dtype=np.int64)
    padded[:, 0] = -1
    padded[:, 1:-1] = dividers
    padded[:, -1] = subdivisions + objective_count - 1

    # noinspection PyTypeChecker
    return (  # pyright: ignore[reportReturnType]
        (np.diff(padded, axis=1) - 1) / subdivisions
    )


def _build_reference_points(
    objective_count: int,
    subdivisions: int,
    inner_subdivisions: int | None,
    inner_scale: float,
) -> FloatMatrix:
    outer_points = _generate_reference_points(objective_count, subdivisions)
    if inner_subdivisions is None:
        return outer_points

    inner_raw = _generate_reference_points(objective_count, inner_subdivisions)
    center = 1.0 / objective_count
    inner_points = center + inner_scale * (inner_raw - center)
    # noinspection PyTypeChecker
    return (  # pyright: ignore[reportReturnType]
        np.vstack([outer_points, inner_points])
    )


def _normalize_reference_directions(ref_points: FloatMatrix) -> FloatMatrix:
    norms: FloatMatrix = (  # pyright: ignore[reportAny]
        np.linalg.norm(ref_points, axis=1, keepdims=True)
    )
    safe_norms = np.where(norms > _NORM_EPSILON, norms, 1.0)
    # noinspection PyTypeChecker
    return ref_points / safe_norms  # pyright: ignore[reportReturnType]


def _extract_minimization_objectives(
    population: Population,
    expected_objectives: int,
    cached_weights: tuple[float, ...] | None,
    cached_min_sign: FloatVector | None,
) -> tuple[FloatMatrix, tuple[float, ...], FloatVector]:
    weights_tuple: tuple[float, ...] | None = cached_weights
    population_size = len(population)
    raw_matrix = np.empty(
        (population_size, expected_objectives),
        dtype=np.float64,
    )

    for row_index, individual in enumerate(population):
        fitness = individual.fitness
        if not fitness.valid:
            raise ValueError(
                "All individuals in population must have valid fitness values."
            )
        values = fitness.values
        weights = fitness.weights
        if (
            len(values) != expected_objectives
            or len(weights) != expected_objectives
        ):
            raise ValueError(
                f"Expected {expected_objectives} objectives/weights, "
                + f"got {len(values)} values and {len(weights)} weights."
            )
        if weights_tuple is None:
            weights_tuple = weights
        elif weights != weights_tuple:
            raise ValueError(
                "All individuals in population must share identical "
                + "fitness weights."
            )
        raw_matrix[row_index] = values

    assert weights_tuple is not None

    if cached_min_sign is None or cached_weights != weights_tuple:
        weights_array = np.asarray(weights_tuple, dtype=np.float64)
        if np.any(weights_array == 0.0):  # pyright: ignore[reportAny]
            raise ValueError("Fitness weights must be non-zero.")
        # noinspection PyTypeChecker
        min_sign: FloatVector = (  # pyright: ignore[reportAssignmentType]
            np.where(weights_array > 0.0, -1.0, 1.0)
        )
    else:
        min_sign = cached_min_sign

    # noinspection PyTypeChecker
    minimized: FloatMatrix = (  # pyright: ignore[reportAssignmentType]
        raw_matrix * min_sign
    )
    return minimized, weights_tuple, min_sign


def _compute_dominance_matrix(
    objectives: FloatMatrix,
    chunk_size: int = _DOMINANCE_CHUNK_SIZE,
) -> BoolMatrix:
    objective_count = objectives.shape[0]
    dominates: BoolMatrix = np.zeros(
        (objective_count, objective_count),
        dtype=np.bool,
    )

    for start in range(0, objective_count, chunk_size):
        end = min(start + chunk_size, objective_count)
        differences = objectives[start:end, None, :] - objectives[None, :, :]
        dominates_chunk: BoolMatrix = (  # pyright: ignore[reportAny]
            np.all(differences <= 0.0, axis=2)
            & np.any(differences < 0.0, axis=2)
        )
        dominates[start:end] = dominates_chunk

    np.fill_diagonal(dominates, False)
    return dominates


def _non_dominated_sort(
    objectives: FloatMatrix,
    max_selected: int | None = None,
) -> list[list[int]]:
    objective_count = objectives.shape[0]
    if objective_count == 0:
        return []

    dominates = _compute_dominance_matrix(objectives)
    domination_counts: IntVector = (  # pyright: ignore[reportAny]
        np.sum(  # pyright: ignore[reportAny]
            dominates,
            axis=0,
        ).astype(np.int64)
    )

    dominated_lists: list[list[int]] = [
        np.flatnonzero(
            dominates[i],  # pyright: ignore[reportAny]
        )
        .astype(np.int64)
        .tolist()
        for i in range(objective_count)
    ]

    fronts: list[list[int]] = []
    current_front: list[int] = (  # pyright: ignore[reportAny]
        np.flatnonzero(
            domination_counts == 0,  # pyright: ignore[reportAny]
        )
        .astype(np.int64)
        .tolist()
    )
    selected_so_far = 0

    while current_front:
        fronts.append(current_front)
        selected_so_far += len(current_front)
        if max_selected is not None and selected_so_far >= max_selected:
            break

        next_front: list[int] = []
        for dominator_index in current_front:
            for dominated_index in dominated_lists[dominator_index]:
                updated_count = (
                    int(
                        domination_counts[  # pyright: ignore[reportAny]
                            dominated_index
                        ]
                    )
                    - 1
                )
                domination_counts[dominated_index] = updated_count
                if updated_count == 0:
                    next_front.append(dominated_index)
        current_front = next_front

    return fronts


def _normalize_objectives(
    objectives: FloatMatrix,
    objective_count: int,
    other_axis_indices: IntMatrix,
) -> FloatMatrix:
    min_objective_vector: FloatVector = (  # pyright: ignore[reportAny]
        np.min(objectives, axis=0)
    )
    # noinspection PyTypeChecker
    translated: FloatMatrix = (  # pyright: ignore[reportAssignmentType]
        objectives - min_objective_vector
    )

    scaled = translated * _ASF_PENALTY_SCALE
    extreme_points = np.zeros(
        (objective_count, objective_count),
        dtype=np.float64,
    )

    for objective_index in range(objective_count):
        other_columns = scaled[:, other_axis_indices[objective_index]]
        other_max: FloatVector = (  # pyright: ignore[reportAny]
            np.max(other_columns, axis=1)
        )
        asf_values = np.maximum(translated[:, objective_index], other_max)
        extreme_points[objective_index] = translated[
            int(np.argmin(asf_values))
        ]

    intercepts = _compute_intercepts(
        extreme_points=extreme_points,
        translated_active=translated,
        objective_count=objective_count,
    )
    # noinspection PyTypeChecker
    return translated / intercepts  # pyright: ignore[reportReturnType]


def _compute_intercepts(
    extreme_points: FloatMatrix,
    translated_active: FloatMatrix,
    objective_count: int,
) -> FloatVector:
    extreme_max: FloatVector = (  # pyright: ignore[reportAny]
        np.max(extreme_points, axis=0)
    )
    active_max: FloatVector = (  # pyright: ignore[reportAny]
        np.max(translated_active, axis=0)
    )
    # noinspection PyTypeChecker
    fallback: FloatVector = (  # pyright: ignore[reportAssignmentType]
        np.where(
            extreme_max > _INTERCEPT_EPSILON,
            extreme_max,
            np.where(active_max > _INTERCEPT_EPSILON, active_max, 1.0),
        )
    )

    ones = np.ones(objective_count, dtype=np.float64)
    try:
        solved = np.linalg.solve(extreme_points, ones)
    except np.linalg.LinAlgError:
        return fallback

    if not np.all(np.isfinite(solved)) or np.any(solved <= _INTERCEPT_EPSILON):
        return fallback

    # noinspection PyTypeChecker
    intercepts: FloatVector = (  # pyright: ignore[reportAssignmentType]
        1.0 / solved
    )
    if not np.all(np.isfinite(intercepts)) or np.any(
        intercepts <= _INTERCEPT_EPSILON
    ):
        return fallback

    return intercepts


def _niching_selection(
    normalized_objectives: FloatMatrix,
    selected_count: int,
    critical_front: list[int],
    unit_reference_points: FloatMatrix,
    remaining_to_select_count: int,
    rng: np.random.Generator,
) -> list[int]:
    reference_count = unit_reference_points.shape[0]
    objective_norm_squared: FloatVector = (  # pyright: ignore[reportAny]
        np.einsum("ij,ij->i", normalized_objectives, normalized_objectives)
    )
    projection_length = normalized_objectives @ unit_reference_points.T
    distances_squared = np.maximum(
        objective_norm_squared[:, None]
        - projection_length * projection_length,
        0.0,
    )
    nearest_reference: IntVector = (  # pyright: ignore[reportAny]
        np.argmin(  # pyright: ignore[reportAny]
            distances_squared,
            axis=1,
        ).astype(np.int64)
    )

    selected_reference = nearest_reference[:selected_count]
    critical_reference = nearest_reference[selected_count:]

    niche_counts = np.bincount(
        selected_reference,
        minlength=reference_count,
    ).astype(np.int64)

    candidates_by_reference: list[list[int]] = [
        [] for _ in range(reference_count)
    ]
    for (
        local_position,
        reference_raw,  # pyright: ignore[reportAny]
    ) in enumerate(critical_reference.tolist()):  # pyright: ignore[reportAny]
        candidates_by_reference[
            int(reference_raw)  # pyright: ignore[reportAny]
        ].append(local_position)

    active_references: list[int] = []
    active_positions: dict[int, int] = {}
    for reference_index, pool in enumerate(candidates_by_reference):
        if pool:
            active_positions[reference_index] = len(active_references)
            active_references.append(reference_index)

    if not active_references:
        return []

    def _collect_min_rho_references() -> tuple[int, list[int]]:
        min_value = min(
            int(niche_counts[reference])  # pyright: ignore[reportAny]
            for reference in active_references
        )
        tied_references = [
            reference
            for reference in active_references
            if int(
                niche_counts[  # pyright: ignore[reportAny]
                    reference
                ]
            )
            == min_value
        ]
        return min_value, tied_references

    current_min_rho, min_rho_references = _collect_min_rho_references()
    chosen_global_indices: list[int] = []

    while (
        len(chosen_global_indices) < remaining_to_select_count
        and active_references
    ):
        if not min_rho_references:
            current_min_rho, min_rho_references = _collect_min_rho_references()

        reference_slot = int(rng.integers(0, len(min_rho_references)))
        chosen_reference = min_rho_references[reference_slot]
        pool = candidates_by_reference[chosen_reference]

        if current_min_rho == 0:
            best_index = 0
            best_distance_squared = float(
                distances_squared[  # pyright: ignore[reportAny]
                    selected_count + pool[0], chosen_reference
                ]
            )
            for index in range(1, len(pool)):
                distance_squared = float(
                    distances_squared[  # pyright: ignore[reportAny]
                        selected_count + pool[index], chosen_reference
                    ]
                )
                if distance_squared < best_distance_squared - _TIE_EPSILON:
                    best_distance_squared = distance_squared
                    best_index = index
                elif (
                    abs(distance_squared - best_distance_squared)
                    <= _TIE_EPSILON
                    and rng.random() < 0.5
                ):
                    best_index = index
            pick_index = best_index
        else:
            pick_index = int(rng.integers(0, len(pool)))

        picked_local = pool[pick_index]
        last_pool_index = len(pool) - 1
        if pick_index != last_pool_index:
            pool[pick_index] = pool[last_pool_index]
        _ = pool.pop()

        niche_counts[chosen_reference] = (
            int(
                niche_counts[chosen_reference]  # pyright: ignore[reportAny]
            )
            + 1
        )
        chosen_global_indices.append(critical_front[picked_local])

        if not pool:
            removed_position = active_positions.pop(chosen_reference)
            last_active_reference = active_references.pop()
            if removed_position < len(active_references):
                active_references[removed_position] = last_active_reference
                active_positions[last_active_reference] = removed_position

        _ = min_rho_references.pop(reference_slot)

    return chosen_global_indices


def sel_spea_2() -> SelectionOperator:
    typed_sel_spea_2 = cast(
        Callable[[Population, int], Population],
        tools.selSPEA2,
    )

    def sel_spea_2_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_spea_2(population, k)

    return sel_spea_2_wrapper


def sel_random(
    seed: int | np.random.Generator | None = None,
    *,
    replace: bool = True,
) -> SelectionOperator:
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )

    def sel_random_wrapper(population: Population, k: int) -> Population:
        population_size = len(population)
        if population_size == 0:
            raise ValueError("'population' must not be empty.")
        if k <= 0:
            raise ValueError(f"'k' must be positive. Got: {k}")
        if not replace and k > population_size:
            raise ValueError(
                f"'k' ({k}) cannot exceed population size ({population_size}) "
                + "when sampling without replacement."
            )

        if replace:
            indices = rng.integers(0, population_size, size=k, dtype=np.int64)
        else:
            indices = rng.choice(
                population_size,
                size=k,
                replace=False,
            ).astype(np.int64)

        return [population[int(index)] for index in indices.tolist()]

    return sel_random_wrapper


def sel_best(fitness_attribute: str = "fitness") -> SelectionOperator:
    typed_sel_best = cast(
        Callable[[Population, int, str], Population],
        tools.selBest,
    )

    def sel_best_wrapper(population: Population, k: int) -> Population:
        return typed_sel_best(population, k, fitness_attribute)

    return sel_best_wrapper


def sel_worst(fitness_attribute: str = "fitness") -> SelectionOperator:
    typed_sel_worst = cast(
        Callable[[Population, int, str], Population],
        tools.selWorst,
    )

    def sel_worst_wrapper(population: Population, k: int) -> Population:
        return typed_sel_worst(population, k, fitness_attribute)

    return sel_worst_wrapper


def sel_double_tournament(
    fitness_size: int,
    parsimony_size: float,
    fitness_first: bool,
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_double_tournament = cast(
        Callable[[Population, int, int, float, bool, str], Population],
        tools.selDoubleTournament,
    )

    def sel_double_tournament_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_double_tournament(
            population,
            k,
            fitness_size,
            parsimony_size,
            fitness_first,
            fitness_attribute,
        )

    return sel_double_tournament_wrapper


def sel_stochastic_universal_sampling(
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_stochastic_universal_sampling = cast(
        Callable[[Population, int, str], Population],
        tools.selStochasticUniversalSampling,
    )

    def sel_stochastic_universal_sampling_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_stochastic_universal_sampling(
            population,
            k,
            fitness_attribute,
        )

    return sel_stochastic_universal_sampling_wrapper


def sel_tournament_dcd() -> SelectionOperator:
    typed_sel_tournament_dcd = cast(
        Callable[[Population, int], Population],
        tools.selTournamentDCD,
    )

    def sel_tournament_dcd_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_tournament_dcd(population, k)

    return sel_tournament_dcd_wrapper


def sel_lexicase() -> SelectionOperator:
    typed_sel_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selLexicase,
    )

    def sel_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_lexicase(population, k)

    return sel_lexicase_wrapper


def sel_epsilon_lexicase(epsilon: float) -> SelectionOperator:
    typed_sel_epsilon_lexicase = cast(
        Callable[[Population, int, float], Population],
        tools.selEpsilonLexicase,
    )

    def sel_epsilon_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_epsilon_lexicase(population, k, epsilon)

    return sel_epsilon_lexicase_wrapper


def sel_automatic_epsilon_lexicase() -> SelectionOperator:
    typed_sel_automatic_epsilon_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selAutomaticEpsilonLexicase,
    )

    def sel_automatic_epsilon_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_automatic_epsilon_lexicase(population, k)

    return sel_automatic_epsilon_lexicase_wrapper
