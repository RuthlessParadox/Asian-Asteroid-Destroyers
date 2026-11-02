from collections.abc import Callable, Sequence

import numpy as np
import pytest

from asian_asteroid_destroyers.evokit import (
    Individual,
    Statistics,
    clear_fitness_cache,
    clear_individual_cache,
    get_individual_class,
)


@pytest.fixture(autouse=True)
def cleanup_caches(request: pytest.FixtureRequest) -> None:
    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


@pytest.fixture
def individual_factory() -> Callable[[tuple[float, ...]], Individual]:
    def create_individual(fitness_values: tuple[float, ...]) -> Individual:
        weights = tuple(1.0 for _ in fitness_values)
        individual_class = get_individual_class(weights)
        individual = individual_class()
        individual.fitness.values = fitness_values
        return individual

    return create_individual


def test_statistics_default_initialization(
    individual_factory: Callable[[tuple[float, ...]], Individual],
) -> None:
    stats = Statistics()
    individual = individual_factory((3.5, 4.5))
    assert stats.key(individual) == (3.5, 4.5)
    assert stats.index == 0


def test_statistics_custom_initialization(
    individual_factory: Callable[[tuple[float, ...]], Individual],
) -> None:
    def custom_key(_individual: Individual) -> tuple[float, ...]:
        return 10.0, 20.0, 30.0

    stats = Statistics(key=custom_key, index=2)
    individual = individual_factory((1.0,))
    assert stats.key(individual) == (10.0, 20.0, 30.0)
    assert stats.index == 2


def test_compile_happy_path(
    individual_factory: Callable[[tuple[float, ...]], Individual],
) -> None:
    population: Sequence[Individual] = [
        individual_factory((1.0,)),
        individual_factory((2.0,)),
        individual_factory((3.0,)),
        individual_factory((4.0,)),
        individual_factory((5.0,)),
    ]
    stats = Statistics(index=0)
    result = stats.compile(population)
    assert result["max_fitness"] == 5.0
    assert result["min_fitness"] == 1.0
    assert result["mean_fitness"] == 3.0
    assert result["median_fitness"] == 3.0
    assert pytest.approx(result["std_fitness"]) == np.std(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )


def test_compile_with_custom_index(
    individual_factory: Callable[[tuple[float, ...]], Individual],
) -> None:
    population: Sequence[Individual] = [
        individual_factory((1.0, 100.0)),
        individual_factory((2.0, 200.0)),
        individual_factory((3.0, 300.0)),
    ]
    stats = Statistics(index=1)
    result = stats.compile(population)
    assert result["max_fitness"] == 300.0
    assert result["min_fitness"] == 100.0
    assert result["mean_fitness"] == 200.0
    assert result["median_fitness"] == 200.0


def test_compile_raises_value_error_for_empty_population() -> None:
    stats = Statistics()
    empty_population: list[Individual] = []
    with pytest.raises(
        ValueError, match="Cannot compile statistics from an empty population."
    ):
        _ = stats.compile(empty_population)


def test_compile_raises_index_error_out_of_bounds() -> None:
    individual_1 = get_individual_class((1.0, 1.0))()
    individual_1.fitness.values = (1.0, 2.0)

    individual_2 = get_individual_class((1.0,))()
    individual_2.fitness.values = (1.0,)

    population: Sequence[Individual] = [individual_1, individual_2]
    stats = Statistics(index=1)

    expected_message = (
        "Target index 1 is out of bounds for one or more "
        "individuals in the population."
    )
    with pytest.raises(IndexError, match=expected_message):
        _ = stats.compile(population)
