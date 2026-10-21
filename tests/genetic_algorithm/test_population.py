import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    fitness as fitness_module,
)
from asian_asteroid_destroyers.genetic_algorithm.fitness import (
    Fitness,
    clear_fitness_cache,
    get_fitness_class,
)
from asian_asteroid_destroyers.genetic_algorithm.individual import Individual
from asian_asteroid_destroyers.genetic_algorithm.population import (
    reweight_population,
)
from asian_asteroid_destroyers.genetic_algorithm.types import Population


@pytest.fixture(autouse=True)
def cleanup_fitness_cache(request: pytest.FixtureRequest) -> None:
    request.addfinalizer(clear_fitness_cache)


def test_reweight_population_migrates_valid_fitness() -> None:
    initial_weights = (1.0,)
    initial_class = get_fitness_class(initial_weights)
    initial_fitness = initial_class()
    initial_fitness.values = (10.5,)

    individual = Individual()
    individual.fitness = initial_fitness
    population: Population = [individual]

    new_weights = (-1.0,)
    expected_class = get_fitness_class(new_weights)

    reweight_population(population, new_weights)

    assert isinstance(individual.fitness, expected_class)
    assert individual.fitness.valid is True
    assert individual.fitness.values == (10.5,)


def test_reweight_population_handles_uninitialized_fitness() -> None:
    initial_weights = (1.0,)
    initial_class = get_fitness_class(initial_weights)
    initial_fitness = initial_class()

    individual = Individual()
    individual.fitness = initial_fitness
    population: Population = [individual]

    new_weights = (1.0, -1.0)
    expected_class = get_fitness_class(new_weights)

    reweight_population(population, new_weights)

    assert isinstance(individual.fitness, expected_class)
    assert individual.fitness.valid is False


def test_reweight_population_handles_corrupted_fitness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_weights = (1.0,)
    initial_class = get_fitness_class(initial_weights)
    initial_fitness = initial_class()

    def valid_stub(_self: Fitness) -> bool:
        return True

    monkeypatch.setattr(
        initial_class,
        "valid",
        property(fget=valid_stub),
    )

    def mock_values_error() -> tuple[float, ...]:
        raise AttributeError("Corrupted values attribute")

    def getter_stub(_self: Fitness) -> tuple[float, ...]:
        return mock_values_error()

    monkeypatch.setattr(
        initial_class,
        "values",
        property(fget=getter_stub),
    )

    individual = Individual()
    individual.fitness = initial_fitness
    population: Population = [individual]

    new_weights = (5.0,)
    expected_class = get_fitness_class(new_weights)

    reweight_population(population, new_weights)

    assert isinstance(individual.fitness, expected_class)
    assert individual.fitness.valid is False


def test_reweight_population_empty_population() -> None:
    population: Population = []
    new_weights = (1.0, 1.0)

    reweight_population(population, new_weights)

    clean_weight_string = Fitness.get_clean_weight_string(new_weights)
    expected_class_name = f"DynamicFitness_{clean_weight_string}"
    assert hasattr(fitness_module, expected_class_name)
