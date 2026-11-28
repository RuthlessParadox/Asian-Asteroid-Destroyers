from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pytest

from asian_asteroid_destroyers.evokit import (
    clear_fitness_cache,
    clear_individual_cache,
    get_individual_class,
    sel_random,
)

if TYPE_CHECKING:
    from asian_asteroid_destroyers.evokit import Individual, Population


@pytest.fixture(autouse=True)
def reset_caches(request: pytest.FixtureRequest) -> None:
    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


def _make_population(
    size: int,
    weights: tuple[float, ...] = (1.0,),
) -> list[Individual]:
    individual_class = get_individual_class(weights)
    population: list[Individual] = []
    for index in range(size):
        individual = individual_class([float(index)])
        individual.fitness.values = tuple(float(index) for _ in weights)
        population.append(individual)
    return population


def test_sel_random_empty_population_raises() -> None:
    selector = sel_random(seed=42)
    empty_population: Population = []

    with pytest.raises(ValueError, match="population.*must not be empty"):
        _ = selector(empty_population, 1)


@pytest.mark.parametrize("k", [0, -1, -10])
def test_sel_random_non_positive_k_raises(k: int) -> None:
    selector = sel_random(seed=42)
    population = _make_population(5)

    with pytest.raises(ValueError, match="'k' must be positive"):
        _ = selector(population, k)


def test_sel_random_without_replacement_k_exceeds_population_raises() -> None:
    selector = sel_random(seed=42, replace=False)
    population = _make_population(4)

    with pytest.raises(ValueError, match="cannot exceed population size"):
        _ = selector(population, 5)


def test_sel_random_with_replacement_allows_k_exceeding_population() -> None:
    selector = sel_random(seed=42, replace=True)
    population = _make_population(3)

    selected = selector(population, 10)

    assert len(selected) == 10
    population_set = set(map(id, population))
    assert all(id(individual) in population_set for individual in selected)


def test_sel_random_without_replacement_no_duplicates() -> None:
    selector = sel_random(seed=123, replace=False)
    population = _make_population(6)

    selected = selector(population, 6)

    assert len(selected) == 6
    assert len({id(individual) for individual in selected}) == 6
    assert {individual[0] for individual in selected} == {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    }


def test_sel_random_seed_reproducibility_with_int() -> None:
    population = _make_population(10)

    first = sel_random(seed=2026, replace=True)(population, 7)
    second = sel_random(seed=2026, replace=True)(population, 7)

    assert [id(individual) for individual in first] == [
        id(individual) for individual in second
    ]


def test_sel_random_seed_reproducibility_with_generator() -> None:
    population = _make_population(10)
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)

    selector_a = sel_random(seed=rng_a, replace=False)
    selector_b = sel_random(seed=rng_b, replace=False)

    assert [id(individual) for individual in selector_a(population, 4)] == [
        id(individual) for individual in selector_b(population, 4)
    ]


def test_sel_random_returns_original_individual_references() -> None:
    population = _make_population(5)
    selector = sel_random(seed=7, replace=True)

    selected = selector(population, 8)

    for chosen in selected:
        gene_value = int(chosen[0])
        assert chosen is population[gene_value]


def test_sel_random_approximate_uniformity_with_replacement() -> None:
    population_size = 4
    draws = 4000
    population = _make_population(population_size)
    selector = sel_random(seed=2026, replace=True)

    selected = selector(population, draws)
    counts = Counter(int(individual[0]) for individual in selected)

    expected = draws / population_size
    for index in range(population_size):
        assert abs(counts[index] - expected) < 120


def test_sel_random_accepts_tuple_population() -> None:
    selector = sel_random(seed=11, replace=False)
    population_tuple: tuple[Individual, ...] = tuple(_make_population(5))

    selected = selector(population_tuple, 3)

    assert len(selected) == 3
    assert len({id(individual) for individual in selected}) == 3
    assert all(individual in population_tuple for individual in selected)


def test_sel_random_multi_objective_and_negative_weights() -> None:
    weights = (1.0, -1.0, 0.5)
    population = _make_population(6, weights=weights)
    selector = sel_random(seed=21, replace=False)

    selected = selector(population, 4)

    assert len(selected) == 4
    for individual in selected:
        assert individual.fitness.valid
        assert individual.fitness.weights == weights
        assert len(individual.fitness.values) == len(weights)


def test_sel_random_without_replacement_full_population() -> None:
    selector = sel_random(seed=303, replace=False)
    population = _make_population(5)

    selected = selector(population, len(population))

    assert len(selected) == len(population)
    assert {id(individual) for individual in selected} == {
        id(individual) for individual in population
    }
