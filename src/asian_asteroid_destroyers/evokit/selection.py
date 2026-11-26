from __future__ import annotations

import itertools
import math
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


def sel_nsga_3(
    num_objectives: int = 5,
    subdivisions: int = 4,
    *,
    inner_subdivisions: int | None = None,
    inner_scale: float = 0.5,
    seed: int | np.random.Generator | None = None,
) -> SelectionOperator:
    if num_objectives < 2:
        raise ValueError(
            f"'num_objectives' must be at least 2. Got: {num_objectives}"
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

    ref_points = _build_reference_points(
        num_objectives=num_objectives,
        subdivisions=subdivisions,
        inner_subdivisions=inner_subdivisions,
        inner_scale=inner_scale,
    )
    unit_ref_points = _normalize_ref_directions(ref_points)
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )
    cached_weights: tuple[float, ...] | None = None
    cached_min_sign: FloatVector | None = None

    def sel_nsga_3_wrapper(population: Population, k: int) -> Population:
        nonlocal cached_weights, cached_min_sign

        if not population:
            raise ValueError("'population' must not be empty.")
        if k <= 0 or k > len(population):
            raise ValueError(
                f"'k' must be in [1, {len(population)}]. Got: {k}"
            )

        objectives, cached_weights, cached_min_sign = (
            _extract_minimization_objectives(
                population,
                expected_objectives=num_objectives,
                cached_weights=cached_weights,
                cached_min_sign=cached_min_sign,
            )
        )
        fronts = _non_dominated_sort(objectives, max_selected=k)

        selected_indices: list[int] = []
        front_idx = 0
        while front_idx < len(fronts):
            current_front = fronts[front_idx]
            if len(selected_indices) + len(current_front) <= k:
                selected_indices.extend(current_front)
                front_idx += 1
            else:
                break

        remaining = k - len(selected_indices)
        if remaining == 0:
            return [population[idx] for idx in selected_indices]

        critical_front = fronts[front_idx]
        combined_indices = selected_indices + critical_front
        normalized = _normalize_objectives(
            objectives=objectives,
            fronts=fronts,
            front_limit=front_idx,
            active_indices=combined_indices,
        )
        chosen_from_critical = _niching_selection(
            normalized_objectives=normalized,
            num_selected=len(selected_indices),
            critical_front=critical_front,
            unit_ref_points=unit_ref_points,
            num_to_select=remaining,
            rng=rng,
        )
        selected_indices.extend(chosen_from_critical)
        return [population[idx] for idx in selected_indices]

    return sel_nsga_3_wrapper


def _generate_reference_points(
    num_objectives: int,
    subdivisions: int,
) -> FloatMatrix:
    comb = itertools.combinations(
        range(subdivisions + num_objectives - 1),
        num_objectives - 1,
    )

    dividers = np.fromiter(
        itertools.chain.from_iterable(comb),
        dtype=np.int64,
    ).reshape(-1, num_objectives - 1)

    n_points = dividers.shape[0]
    padded = np.empty((n_points, num_objectives + 1), dtype=np.int64)
    padded[:, 0] = -1
    padded[:, 1:-1] = dividers
    padded[:, -1] = subdivisions + num_objectives - 1

    # noinspection PyTypeChecker
    return (  # pyright: ignore[reportReturnType]
        (np.diff(padded, axis=1) - 1) / subdivisions
    )


def _build_reference_points(
    num_objectives: int,
    subdivisions: int,
    inner_subdivisions: int | None,
    inner_scale: float,
) -> FloatMatrix:
    outer_points = _generate_reference_points(num_objectives, subdivisions)
    if inner_subdivisions is None:
        return outer_points

    inner_raw = _generate_reference_points(num_objectives, inner_subdivisions)
    center = 1.0 / num_objectives
    inner_points = center + inner_scale * (inner_raw - center)
    # noinspection PyTypeChecker
    return (  # pyright: ignore[reportReturnType]
        np.vstack([outer_points, inner_points])
    )


def _normalize_ref_directions(ref_points: FloatMatrix) -> FloatMatrix:
    norms: FloatMatrix = (  # pyright: ignore[reportAny]
        np.linalg.norm(ref_points, axis=1, keepdims=True)
    )
    safe_norms = np.where(norms > 1e-18, norms, 1.0)
    # noinspection PyTypeChecker
    return ref_points / safe_norms  # pyright: ignore[reportReturnType]


def _extract_minimization_objectives(
    population: Population,
    expected_objectives: int,
    cached_weights: tuple[float, ...] | None,
    cached_min_sign: FloatVector | None,
) -> tuple[FloatMatrix, tuple[float, ...], FloatVector]:
    raw_rows: list[tuple[float, ...]] = []
    weights_tuple: tuple[float, ...] | None = cached_weights

    for ind in population:
        fitness = ind.fitness
        if not fitness.valid:
            raise ValueError(
                "All individuals in population must have valid fitness values."
            )
        vals = fitness.values
        w = fitness.weights
        if len(vals) != expected_objectives or len(w) != expected_objectives:
            raise ValueError(
                f"Expected {expected_objectives} objectives/weights, "
                + f"got {len(vals)} values and {len(w)} weights."
            )
        if weights_tuple is None:
            weights_tuple = w
        elif w != weights_tuple:
            raise ValueError(
                "All individuals in population must share identical "
                + "fitness weights."
            )
        raw_rows.append(vals)

    assert weights_tuple is not None
    if cached_min_sign is None or cached_weights != weights_tuple:
        weights_arr = np.asarray(weights_tuple, dtype=np.float64)
        if bool(np.any(weights_arr == 0.0)):  # pyright: ignore[reportAny]
            raise ValueError("Fitness weights must be non-zero.")
        # noinspection PyTypeChecker
        min_sign: FloatVector = (  # pyright: ignore[reportAssignmentType]
            np.where(weights_arr > 0.0, -1.0, 1.0)
        )
    else:
        min_sign = cached_min_sign

    raw_matrix = np.asarray(raw_rows, dtype=np.float64)
    # Maximize (+w) -> -val, minimize (-w) -> +val.
    # noinspection PyTypeChecker
    minimized: FloatMatrix = (  # pyright: ignore[reportAssignmentType]
        raw_matrix * min_sign
    )
    return minimized, weights_tuple, min_sign


def _non_dominated_sort(
    objectives: FloatMatrix,
    max_selected: int | None = None,
) -> list[list[int]]:
    n = objectives.shape[0]
    if n == 0:
        return []

    le = objectives[:, None, :] <= objectives[None, :, :]
    lt = objectives[:, None, :] < objectives[None, :, :]
    dominates: BoolMatrix = (  # pyright: ignore[reportAny]
        np.all(le, axis=2) & np.any(lt, axis=2)
    )
    np.fill_diagonal(dominates, False)

    domination_count: IntVector = (  # pyright: ignore[reportAny]
        np.sum(  # pyright: ignore[reportAny]
            dominates,
            axis=0,
        ).astype(np.int64)
    )
    dominated_lists: list[list[int]] = [[] for _ in range(n)]
    for p in range(n):
        dominated_lists[p] = [
            int(q)  # pyright: ignore[reportAny]
            for q in np.flatnonzero(dominates[p])  # pyright: ignore[reportAny]
        ]

    fronts: list[list[int]] = []
    current_front = [
        int(i)  # pyright: ignore[reportAny]
        for i in np.flatnonzero(  # pyright: ignore[reportAny]
            domination_count == 0,  # pyright: ignore[reportAny]
        )
    ]
    selected_so_far = 0

    while current_front:
        fronts.append(current_front)
        selected_so_far += len(current_front)
        if max_selected is not None and selected_so_far >= max_selected:
            break

        next_front: list[int] = []
        for p in current_front:
            for q in dominated_lists[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current_front = next_front

    return fronts


def _normalize_objectives(
    objectives: FloatMatrix,
    fronts: list[list[int]],
    front_limit: int,
    active_indices: list[int],
) -> FloatMatrix:
    ideal_pool_list: list[int] = []
    for f_idx in range(front_limit + 1):
        ideal_pool_list.extend(fronts[f_idx])
    ideal_pool = objectives[ideal_pool_list]
    z_min: FloatVector = (  # pyright: ignore[reportAny]
        np.min(ideal_pool, axis=0)
    )

    # Translated objectives in minimization space: f'_i = f_i - z_min.
    # noinspection PyTypeChecker
    translated_all: FloatMatrix = (  # pyright: ignore[reportAssignmentType]
        objectives[active_indices] - z_min
    )
    num_objs = objectives.shape[1]

    # Scalarizing achievement function (ASF) to locate extreme points along each axis.
    extreme_points = np.zeros((num_objs, num_objs), dtype=np.float64)
    for obj_idx in range(num_objs):
        w_vec = np.full(num_objs, 1e-6, dtype=np.float64)
        w_vec[obj_idx] = 1.0
        asf_vals: FloatVector = (  # pyright: ignore[reportAny]
            np.max(translated_all / w_vec, axis=1)
        )
        best_local_idx = int(np.argmin(asf_vals))
        extreme_points[obj_idx] = translated_all[best_local_idx]

    intercepts = _compute_intercepts(
        extreme_points=extreme_points,
        translated_active=translated_all,
        num_objs=num_objs,
    )
    # noinspection PyTypeChecker
    return translated_all / intercepts  # pyright: ignore[reportReturnType]


def _compute_intercepts(
    extreme_points: FloatMatrix,
    translated_active: FloatMatrix,
    num_objs: int,
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
            extreme_max > 1e-12,
            extreme_max,
            np.where(active_max > 1e-12, active_max, 1.0),
        )
    )

    ones = np.ones(num_objs, dtype=np.float64)
    try:
        # Solve extreme_points * x = 1 where x_j = 1 / intercept_j.
        x = np.linalg.solve(extreme_points, ones)
    except np.linalg.LinAlgError:
        return fallback

    if not np.all(np.isfinite(x)) or np.any(x <= 1e-12):
        return fallback

    # noinspection PyTypeChecker
    intercepts: FloatVector = 1.0 / x  # pyright: ignore[reportAssignmentType]
    if not np.all(np.isfinite(intercepts)) or np.any(intercepts <= 1e-12):
        return fallback

    return intercepts


def _assign_nearest_reference_points(
    normalized_objectives: FloatMatrix,
    unit_ref_points: FloatMatrix,
) -> IntVector:
    # For unit direction u, perpendicular distance squared is ||s||^2 - (s . u)^2.
    obj_norm_sq: FloatVector = (  # pyright: ignore[reportAny]
        np.einsum("ij,ij->i", normalized_objectives, normalized_objectives)
    )
    # noinspection PyTypeChecker
    proj_len: FloatMatrix = (  # pyright: ignore[reportAssignmentType]
        normalized_objectives @ unit_ref_points.T
    )
    dist_sq = np.maximum(obj_norm_sq[:, None] - proj_len * proj_len, 0.0)
    return np.argmin(  # pyright: ignore[reportAny]
        dist_sq,
        axis=1,
    ).astype(np.int64)


def _perpendicular_distances_to_ref(
    candidates: FloatMatrix,
    unit_ref: FloatVector,
) -> FloatVector:
    # Exact perpendicular norm ||c - (c . u) * u||_2 on the critical front subset.
    proj_len = candidates @ unit_ref
    projections = proj_len[:, None] * unit_ref[None, :]
    diffs = candidates - projections
    return np.linalg.norm(diffs, axis=1)  # pyright: ignore[reportAny]


def _niching_selection(
    normalized_objectives: FloatMatrix,
    num_selected: int,
    critical_front: list[int],
    unit_ref_points: FloatMatrix,
    num_to_select: int,
    rng: np.random.Generator,
) -> list[int]:
    num_ref = unit_ref_points.shape[0]
    nearest_ref = _assign_nearest_reference_points(
        normalized_objectives,
        unit_ref_points,
    )

    selected_ref = nearest_ref[:num_selected]
    critical_ref = nearest_ref[num_selected:]
    critical_norm_obj = normalized_objectives[num_selected:]

    # Niche count rho_j: number of already-selected individuals associated with ref point j.
    niche_counts = np.bincount(selected_ref, minlength=num_ref).astype(
        np.int64
    )

    # Group critical front local positions by associated reference point.
    candidates_by_ref: list[list[int]] = [[] for _ in range(num_ref)]
    for local_pos in range(critical_ref.shape[0]):
        ref_idx = int(
            critical_ref[local_pos]  # pyright: ignore[reportAny]
        )
        candidates_by_ref[ref_idx].append(local_pos)

    candidate_refs = [
        ref_idx for ref_idx, pool in enumerate(candidates_by_ref) if pool
    ]
    if not candidate_refs:
        return []

    min_rho = min(
        int(niche_counts[r])  # pyright: ignore[reportAny]
        for r in candidate_refs
    )
    min_rho_refs: list[int] = [
        r
        for r in candidate_refs
        if int(niche_counts[r]) == min_rho  # pyright: ignore[reportAny]
    ]

    chosen_global_indices: list[int] = []
    while len(chosen_global_indices) < num_to_select and candidate_refs:
        chosen_ref = min_rho_refs[int(rng.integers(0, len(min_rho_refs)))]
        pool = candidates_by_ref[chosen_ref]

        if int(niche_counts[chosen_ref]) == 0:  # pyright: ignore[reportAny]
            # noinspection PyTypeChecker
            candidate_rows: FloatMatrix = (  # pyright: ignore[reportAssignmentType]
                critical_norm_obj[pool]
            )
            ref_vec: FloatVector = (  # pyright: ignore[reportAny]
                unit_ref_points[chosen_ref]
            )
            dists = _perpendicular_distances_to_ref(candidate_rows, ref_vec)
            min_d = float(np.min(dists))
            ties = [
                pool[i]
                for i in range(dists.shape[0])
                if math.isclose(
                    float(dists[i]),  # pyright: ignore[reportAny]
                    min_d,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]
            if not ties:
                ties = [pool[int(np.argmin(dists))]]
            picked_local = ties[int(rng.integers(0, len(ties)))]
            pool.remove(picked_local)
        else:
            picked_pos = int(rng.integers(0, len(pool)))
            picked_local = pool.pop(picked_pos)

        niche_counts[chosen_ref] += 1
        chosen_global_indices.append(critical_front[picked_local])

        if not pool:
            candidate_refs.remove(chosen_ref)
            min_rho_refs = [r for r in min_rho_refs if r != chosen_ref]
        else:
            min_rho_refs = [r for r in min_rho_refs if r != chosen_ref]

        if not min_rho_refs and candidate_refs:
            min_rho = min(
                int(niche_counts[r])  # pyright: ignore[reportAny]
                for r in candidate_refs
            )
            min_rho_refs = [
                r
                for r in candidate_refs
                if int(
                    niche_counts[r]  # pyright: ignore[reportAny]
                )
                == min_rho
            ]

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


def sel_random() -> SelectionOperator:
    typed_sel_random = cast(
        Callable[[Population, int], Population],
        tools.selRandom,
    )

    def sel_random_wrapper(population: Population, k: int) -> Population:
        return typed_sel_random(population, k)

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
