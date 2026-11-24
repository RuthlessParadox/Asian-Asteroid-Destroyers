from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Callable, cast

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
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed)
    )

    def sel_nsga_3_wrapper(population: Population, k: int) -> Population:
        if not population:
            raise ValueError("'population' must not be empty.")
        if k <= 0 or k > len(population):
            raise ValueError(
                f"'k' must be in [1, {len(population)}]. Got: {k}"
            )

        objectives = _extract_minimization_objectives(
            population,
            expected_objectives=num_objectives,
        )
        fronts = _non_dominated_sort(objectives)

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
        if remaining > 0:
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
                selected_indices=selected_indices,
                critical_front=critical_front,
                ref_points=ref_points,
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


def _extract_minimization_objectives(
    population: Population,
    expected_objectives: int,
) -> FloatMatrix:
    raw_rows: list[tuple[float, ...]] = []
    weights_tuple: tuple[float, ...] | None = None

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
                "All individuals in population must share identical fitness weights."
            )
        raw_rows.append(vals)

    assert weights_tuple is not None
    weights_arr = np.asarray(weights_tuple, dtype=np.float64)
    if np.any(weights_arr == 0.0):  # pyright: ignore[reportAny]
        raise ValueError("Fitness weights must be non-zero.")

    # Convert to minimization space: maximize (+w) -> -val, minimize (-w) -> +val.
    min_sign = np.where(weights_arr > 0.0, -1.0, 1.0)
    # noinspection PyTypeChecker
    return (  # pyright: ignore[reportReturnType]
        np.asarray(raw_rows, dtype=np.float64) * min_sign
    )


def _non_dominated_sort(objectives: FloatMatrix) -> list[list[int]]:
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
        for q in np.flatnonzero(dominates[p]):  # pyright: ignore[reportAny]
            dominated_lists[p].append(int(q))  # pyright: ignore[reportAny]

    fronts: list[list[int]] = []
    current_front = [
        int(i)  # pyright: ignore[reportAny]
        for i in np.flatnonzero(  # pyright: ignore[reportAny]
            domination_count == 0,  # pyright: ignore[reportAny]
        )
    ]

    while current_front:
        fronts.append(current_front)
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
    # Ideal point z_min is computed across fronts F_1 .. F_l (0 .. front_limit).
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


def _associate_to_reference_points(
    normalized_objectives: FloatMatrix,
    ref_points: FloatMatrix,
) -> tuple[IntVector, FloatVector]:
    # Perpendicular distance d_perp(s, w) = ||s - ((s . w) / ||w||^2) * w||_2
    ref_norms_sq: FloatVector = (  # pyright: ignore[reportAny]
        np.sum(ref_points * ref_points, axis=1)
    )
    safe_norms_sq = np.where(ref_norms_sq > 1e-18, ref_norms_sq, 1.0)

    # normalized_objectives: (N, M), ref_points: (R, M)
    # dot_prod: (N, R)
    dot_prod = normalized_objectives @ ref_points.T
    proj_scale = dot_prod / safe_norms_sq[None, :]  # (N, R)

    # projections: (N, R, M)
    projections = proj_scale[:, :, None] * ref_points[None, :, :]
    diffs = normalized_objectives[:, None, :] - projections
    distances: FloatMatrix = (  # pyright: ignore[reportAny]
        np.linalg.norm(diffs, axis=2)
    )  # (N, R)

    nearest_ref: IntVector = (  # pyright: ignore[reportAny]
        np.argmin(  # pyright: ignore[reportAny]
            distances,
            axis=1,
        ).astype(np.int64)
    )
    # noinspection PyTypeChecker
    nearest_dist: FloatVector = (  # pyright: ignore[reportAssignmentType]
        distances[np.arange(len(nearest_ref)), nearest_ref].astype(np.float64)
    )
    return nearest_ref, nearest_dist


def _niching_selection(
    normalized_objectives: FloatMatrix,
    selected_indices: list[int],
    critical_front: list[int],
    ref_points: FloatMatrix,
    num_to_select: int,
    rng: np.random.Generator,
) -> list[int]:
    num_selected = len(selected_indices)
    num_ref = ref_points.shape[0]

    nearest_ref, nearest_dist = _associate_to_reference_points(
        normalized_objectives,
        ref_points,
    )

    selected_ref = nearest_ref[:num_selected]
    critical_ref = nearest_ref[num_selected:]
    critical_dist = nearest_dist[num_selected:]

    # Niche count rho_j: number of already-selected individuals associated with ref point j.
    niche_counts = np.bincount(selected_ref, minlength=num_ref).astype(
        np.int64
    )

    # Group critical front local positions by associated reference point.
    candidates_by_ref: list[list[int]] = [[] for _ in range(num_ref)]
    for (
        local_pos,
        ref_idx,  # pyright: ignore[reportAny]
    ) in enumerate(critical_ref):
        candidates_by_ref[
            int(ref_idx)  # pyright: ignore[reportAny]
        ].append(local_pos)

    chosen_global_indices: list[int] = []
    active_refs = set(range(num_ref))

    while len(chosen_global_indices) < num_to_select and active_refs:
        min_rho = min(
            int(niche_counts[r])  # pyright: ignore[reportAny]
            for r in active_refs
        )
        min_rho_refs = [
            r
            for r in active_refs
            if int(niche_counts[r]) == min_rho  # pyright: ignore[reportAny]
        ]

        chosen_ref = int(min_rho_refs[int(rng.integers(0, len(min_rho_refs)))])
        pool = candidates_by_ref[chosen_ref]

        if not pool:
            active_refs.remove(chosen_ref)
            continue

        if int(niche_counts[chosen_ref]) == 0:  # pyright: ignore[reportAny]
            # Pick candidate with minimum perpendicular distance to this reference line.
            # Duplicate/coincident individuals (e.g. distance == 0.0) are collected as ties
            # and broken reproducibly via rng.
            dists = critical_dist[pool]
            min_d = float(np.min(dists))
            ties = [
                pool[i]
                for (
                    i,
                    d,  # pyright: ignore[reportAny]
                ) in enumerate(dists)
                if math.isclose(
                    float(d),  # pyright: ignore[reportAny]
                    min_d,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]
            if not ties:
                ties = [pool[int(np.argmin(dists))]]
            picked_local = ties[int(rng.integers(0, len(ties)))]
        else:
            # Randomly pick one candidate associated with this reference point.
            picked_local = pool[int(rng.integers(0, len(pool)))]

        pool.remove(picked_local)
        if not pool:
            active_refs.remove(chosen_ref)

        niche_counts[chosen_ref] += 1
        chosen_global_indices.append(critical_front[picked_local])

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
