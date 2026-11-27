from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from typing_extensions import override

from asian_asteroid_destroyers.evokit import (
    Fitness,
    Individual,
    clear_fitness_cache,
    clear_individual_cache,
    get_individual_class,
    sel_nsga_3,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from asian_asteroid_destroyers.evokit import Population


@pytest.fixture(autouse=True)
def reset_caches(request: pytest.FixtureRequest) -> None:
    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


def _make_population(
    values_list: Sequence[tuple[float, ...]],
    weights: tuple[float, ...],
) -> list[Individual]:
    individual_class: type[Individual] = get_individual_class(weights)
    population: list[Individual] = []
    for index, values in enumerate(values_list):
        individual = individual_class([float(index)])
        individual.fitness.values = values
        population.append(individual)
    return population


class TestSelNsga3Validation:
    @pytest.mark.parametrize("num_objectives", [1, 0, -2])
    def test_invalid_num_objectives_raises(self, num_objectives: int) -> None:
        with pytest.raises(ValueError, match="num_objectives"):
            _ = sel_nsga_3(num_objectives=num_objectives)

    @pytest.mark.parametrize("subdivisions", [0, -1])
    def test_invalid_subdivisions_raises(self, subdivisions: int) -> None:
        with pytest.raises(ValueError, match="subdivisions"):
            _ = sel_nsga_3(subdivisions=subdivisions)

    @pytest.mark.parametrize("inner_subdivisions", [0, -3])
    def test_invalid_inner_subdivisions_raises(
        self,
        inner_subdivisions: int,
    ) -> None:
        with pytest.raises(ValueError, match="inner_subdivisions"):
            _ = sel_nsga_3(inner_subdivisions=inner_subdivisions)

    @pytest.mark.parametrize("inner_scale", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_inner_scale_raises(self, inner_scale: float) -> None:
        with pytest.raises(ValueError, match="inner_scale"):
            _ = sel_nsga_3(inner_scale=inner_scale)

    def test_empty_population_raises(self) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=2, seed=42)
        empty_pop: Population = []
        with pytest.raises(ValueError, match="population"):
            _ = selector(empty_pop, 1)

    @pytest.mark.parametrize("k", [0, -1, 4])
    def test_invalid_k_raises(self, k: int) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=2, seed=42)
        pop = _make_population(
            [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)],
            weights=(-1.0, -1.0),
        )
        with pytest.raises(ValueError, match="'k'"):
            _ = selector(pop, k)

    def test_invalid_fitness_values_raises(self) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=2, seed=42)
        ind_cls = get_individual_class((-1.0, -1.0))
        unevaluated = ind_cls([0.0])
        with pytest.raises(ValueError, match="valid fitness"):
            _ = selector([unevaluated], 1)

    def test_mismatched_objective_count_raises(self) -> None:
        selector = sel_nsga_3(num_objectives=3, subdivisions=2, seed=42)
        pop = _make_population([(1.0, 2.0)], weights=(-1.0, -1.0))
        with pytest.raises(ValueError, match="Expected 3 objectives"):
            _ = selector(pop, 1)

    def test_inconsistent_weights_across_population_raises(self) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=2, seed=42)
        pop_a = _make_population([(1.0, 2.0)], weights=(-1.0, -1.0))
        pop_b = _make_population([(2.0, 1.0)], weights=(-1.0, 1.0))
        combined: list[Individual] = [*pop_a, *pop_b]
        with pytest.raises(ValueError, match="identical fitness weights"):
            _ = selector(combined, 1)

    class _ZeroWeightFitness(Fitness):
        weights: tuple[float, ...] = (-1.0, 0.0)

        def __init__(self, raw_values: tuple[float, ...] = (1.0, 2.0)) -> None:
            super().__init__()
            object.__setattr__(self, "_raw_values", raw_values)

        @property
        def values(  # pyright: ignore[reportImplicitOverride]
            self,
        ) -> tuple[float, ...]:
            raw: tuple[float, ...] = (  # pyright: ignore[reportAny]
                object.__getattribute__(self, "_raw_values")
            )
            assert isinstance(raw, tuple)
            return raw

        @values.setter
        def values(  # pyright: ignore[reportIncompatibleVariableOverride]
            self,
            new_values: tuple[float, ...],
        ) -> None:
            object.__setattr__(self, "_raw_values", new_values)

        @property
        @override
        def valid(self) -> bool:
            return True

    def test_zero_weight_raises(self) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=2, seed=42)
        ind = Individual([0.0])
        ind.fitness = TestSelNsga3Validation._ZeroWeightFitness()
        with pytest.raises(ValueError, match="non-zero"):
            _ = selector([ind], 1)


class TestSelNsga3Behavior:
    def test_selects_exact_k_and_prefers_non_dominated(self) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=3, seed=123)
        # With weights (-1.0, -1.0), lower values are better (minimization).
        # First two are non-dominated; third is dominated by both.
        nd1 = (0.1, 0.9)
        nd2 = (0.9, 0.1)
        dominated = (0.8, 0.8)
        pop = _make_population([dominated, nd1, nd2], weights=(-1.0, -1.0))

        selected = selector(pop, 2)
        assert len(selected) == 2
        selected_vals = {ind.fitness.values for ind in selected}
        assert selected_vals == {nd1, nd2}

    def test_respects_maximization_and_mixed_weights(self) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=3, seed=7)
        best_a = (10.0, 1.0)
        best_b = (5.0, 0.5)
        worse = (2.0, 5.0)
        pop = _make_population([worse, best_a, best_b], weights=(1.0, -1.0))

        selected = selector(pop, 2)
        assert len(selected) == 2
        selected_vals = {ind.fitness.values for ind in selected}
        assert selected_vals == {best_a, best_b}

    def test_niching_splits_critical_front_deterministically(self) -> None:
        raw_points = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.33, 0.33, 0.34),
            (0.7, 0.2, 0.1),
        ]
        pop = _make_population(raw_points, weights=(-1.0, -1.0, -1.0))

        sel_a = sel_nsga_3(
            num_objectives=3,
            subdivisions=2,
            inner_subdivisions=1,
            inner_scale=0.5,
            seed=99,
        )
        sel_b = sel_nsga_3(
            num_objectives=3,
            subdivisions=2,
            inner_subdivisions=1,
            inner_scale=0.5,
            seed=99,
        )

        picked_a = sel_a(pop, 4)
        picked_b = sel_b(pop, 4)

        assert len(picked_a) == 4
        assert [ind.fitness.values for ind in picked_a] == [
            ind.fitness.values for ind in picked_b
        ]

    def test_accepts_numpy_generator_seed(self) -> None:
        rng_a = np.random.default_rng(2026)
        rng_b = np.random.default_rng(2026)
        pop = _make_population(
            [
                (1.0, 0.2, 0.8),
                (0.2, 1.0, 0.8),
                (0.8, 0.2, 1.0),
                (0.5, 0.5, 0.5),
                (0.6, 0.4, 0.7),
            ],
            weights=(-1.0, -1.0, -1.0),
        )

        op_a = sel_nsga_3(num_objectives=3, subdivisions=2, seed=rng_a)
        op_b = sel_nsga_3(num_objectives=3, subdivisions=2, seed=rng_b)

        res_a = op_a(pop, 3)
        res_b = op_b(pop, 3)

        assert [ind.fitness.values for ind in res_a] == [
            ind.fitness.values for ind in res_b
        ]

    def test_selecting_full_population_returns_all_individuals(self) -> None:
        selector = sel_nsga_3(num_objectives=2, subdivisions=2, seed=1)
        pop = _make_population(
            [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)],
            weights=(-1.0, -1.0),
        )

        selected = selector(pop, len(pop))
        assert len(selected) == len(pop)
        assert {id(ind) for ind in selected} == {id(ind) for ind in pop}


class TestSelNsga3DegenerateFronts:
    def test_identical_objectives_across_entire_population(self) -> None:
        pop = _make_population(
            [(2.5, 2.5, 2.5)] * 6,
            weights=(-1.0, -1.0, -1.0),
        )
        selector = sel_nsga_3(num_objectives=3, subdivisions=2, seed=11)

        selected = selector(pop, 3)
        assert len(selected) == 3
        assert all(ind.fitness.values == (2.5, 2.5, 2.5) for ind in selected)
        assert len({id(ind) for ind in selected}) == 3

    def test_collinear_front_on_single_line(self) -> None:
        line_points = [(float(t), float(10 - t), 5.0) for t in range(1, 8)]
        pop = _make_population(line_points, weights=(-1.0, -1.0, -1.0))
        selector = sel_nsga_3(num_objectives=3, subdivisions=3, seed=21)

        selected = selector(pop, 4)
        assert len(selected) == 4
        assert len({id(ind) for ind in selected}) == 4

    def test_coplanar_front_with_zero_variance_on_one_objective(self) -> None:
        coplanar = [
            (0.0, 1.0, 0.0),
            (0.25, 0.75, 0.0),
            (0.5, 0.5, 0.0),
            (0.75, 0.25, 0.0),
            (1.0, 0.0, 0.0),
            (0.4, 0.6, 0.0),
        ]
        pop = _make_population(coplanar, weights=(-1.0, -1.0, -1.0))
        selector = sel_nsga_3(num_objectives=3, subdivisions=2, seed=31)

        selected = selector(pop, 3)
        assert len(selected) == 3
        assert len({id(ind) for ind in selected}) == 3

    def test_dominated_duplicates_and_ties_at_boundary(self) -> None:
        pop = _make_population(
            [
                (1.0, 5.0),
                (1.0, 5.0),
                (3.0, 3.0),
                (5.0, 1.0),
                (5.0, 1.0),
                (6.0, 6.0),
            ],
            weights=(-1.0, -1.0),
        )
        selector = sel_nsga_3(num_objectives=2, subdivisions=4, seed=41)

        selected = selector(pop, 4)
        assert len(selected) == 4
        assert all(ind.fitness.values != (6.0, 6.0) for ind in selected)


_finite_coord = st.floats(
    min_value=-1e4,
    max_value=1e4,
    allow_nan=False,
    allow_infinity=False,
)

_nonzero_weight = st.sampled_from([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])


@settings(max_examples=50, deadline=None)
@given(
    num_objectives=st.integers(min_value=2, max_value=4),
    subdivisions=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=10_000),
    data=st.data(),
)
def test_sel_nsga_3_structural_invariants(
    num_objectives: int,
    subdivisions: int,
    seed: int,
    data: st.DataObject,
) -> None:
    clear_individual_cache()
    clear_fitness_cache()

    pop_size = data.draw(st.integers(min_value=1, max_value=25))
    k = data.draw(st.integers(min_value=1, max_value=pop_size))
    weights = tuple(
        data.draw(
            st.lists(
                _nonzero_weight,
                min_size=num_objectives,
                max_size=num_objectives,
            )
        )
    )
    ind_cls: type[Individual] = get_individual_class(weights)

    population: list[Individual] = []
    for idx in range(pop_size):
        vals = tuple(
            data.draw(
                st.lists(
                    _finite_coord,
                    min_size=num_objectives,
                    max_size=num_objectives,
                )
            )
        )
        ind = ind_cls([float(idx)])
        ind.fitness.values = vals
        population.append(ind)

    selector = sel_nsga_3(
        num_objectives=num_objectives,
        subdivisions=subdivisions,
        seed=seed,
    )
    selected = selector(population, k)

    assert len(selected) == k
    selected_ids = [id(ind) for ind in selected]
    assert len(set(selected_ids)) == k

    pop_ids = {id(ind) for ind in population}
    assert set(selected_ids).issubset(pop_ids)

    for ind in selected:
        assert ind.fitness.valid
        assert len(ind.fitness.values) == num_objectives
        assert ind.fitness.weights == weights
