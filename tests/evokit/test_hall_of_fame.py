from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from asian_asteroid_destroyers.evokit import (
    HallOfFame,
    clear_individual_cache,
    get_individual_class,
)

if TYPE_CHECKING:
    from asian_asteroid_destroyers.evokit import Individual


@pytest.fixture(autouse=True)
def cleanup_individual_cache(request: pytest.FixtureRequest) -> None:
    request.addfinalizer(clear_individual_cache)


def _create_individual(
    values: list[float],
    weights: tuple[float, ...],
    fitness_values: tuple[float, ...],
) -> Individual:
    individual_class = get_individual_class(weights)
    individual = individual_class(values)
    individual.fitness.values = fitness_values
    return individual


@pytest.fixture
def populated_hall_of_fame() -> HallOfFame:
    hall_of_fame = HallOfFame()

    weights = (1.0, -1.0)

    individual_1 = _create_individual(
        [1.0],
        weights=weights,
        fitness_values=(5.0, 20.0),
    )
    individual_2 = _create_individual(
        [2.0],
        weights=weights,
        fitness_values=(20.0, 5.0),
    )

    hall_of_fame.update([individual_1, individual_2])
    return hall_of_fame


def test_hall_of_fame_initialization() -> None:
    hall_of_fame = HallOfFame()
    assert len(hall_of_fame) == 0
    assert len(hall_of_fame.items) == 0


def test_hall_of_fame_insert_and_len() -> None:
    hall_of_fame = HallOfFame()
    individual = _create_individual(
        [1.0],
        weights=(1.0, 1.0),
        fitness_values=(10.0, 5.0),
    )
    hall_of_fame.insert(individual)
    assert len(hall_of_fame) == 1
    assert hall_of_fame[0] == individual


def test_hall_of_fame_update_population(
    populated_hall_of_fame: HallOfFame,
) -> None:
    assert len(populated_hall_of_fame) == 1
    assert populated_hall_of_fame[0] == [2.0]


def test_hall_of_fame_remove_and_clear(
    populated_hall_of_fame: HallOfFame,
) -> None:
    assert len(populated_hall_of_fame) == 1

    populated_hall_of_fame.remove(0)
    assert len(populated_hall_of_fame) == 0

    populated_hall_of_fame.clear()
    assert len(populated_hall_of_fame) == 0


def test_hall_of_fame_iteration(populated_hall_of_fame: HallOfFame) -> None:
    items_list = list(populated_hall_of_fame)
    assert len(items_list) == 1
    assert items_list[0] == [2.0]


def test_hall_of_fame_best_empty_raises_index_error() -> None:
    hall_of_fame = HallOfFame()
    with pytest.raises(
        IndexError,
        match="Cannot retrieve an individual from an empty HallOfFame.",
    ):
        _ = hall_of_fame.best


def test_hall_of_fame_multi_objective_tradeoff() -> None:
    hall_of_fame = HallOfFame()
    weights = (1.0, 1.0)

    individual_a = _create_individual([1.0], weights, (10.0, 2.0))
    individual_b = _create_individual([2.0], weights, (2.0, 10.0))

    hall_of_fame.update([individual_a, individual_b])

    assert len(hall_of_fame) == 2
    assert individual_a in hall_of_fame
    assert individual_b in hall_of_fame


def test_hall_of_fame_string_conversion() -> None:
    hall_of_fame = HallOfFame()
    individual = _create_individual(
        [1.0],
        weights=(1.0,),
        fitness_values=(10.0,),
    )
    hall_of_fame.insert(individual)

    assert str(hall_of_fame) == "[[1.0]]"
