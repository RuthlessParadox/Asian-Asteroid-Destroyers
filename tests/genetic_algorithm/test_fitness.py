import pytest
from deap import base

from asian_asteroid_destroyers.genetic_algorithm.fitness import Fitness


@pytest.fixture
def configure_weights(request: pytest.FixtureRequest) -> None:
    def reset_weights() -> None:
        Fitness.weights = ()

    request.addfinalizer(reset_weights)


@pytest.mark.usefixtures("configure_weights")
def test_fitness_initializes_empty() -> None:
    fitness = Fitness()
    assert isinstance(fitness, base.Fitness)
    assert len(fitness.values) == 0
    assert not fitness.valid


@pytest.mark.usefixtures("configure_weights")
def test_fitness_initializes_with_values() -> None:
    Fitness.weights = (1.0, 1.0)
    values: tuple[float, ...] = (1.0, 2.0)

    fitness = Fitness(values)
    assert fitness.values == values


@pytest.mark.usefixtures("configure_weights")
def test_setattr_allows_valid_dimensions() -> None:
    Fitness.weights = (-1.0, 1.0)
    fitness = Fitness()

    valid_values: tuple[float, ...] = (10.0, 20.0)
    fitness.values = valid_values
    assert fitness.values == valid_values


@pytest.mark.usefixtures("configure_weights")
def test_setattr_raises_value_error_on_mismatch() -> None:
    Fitness.weights = (1.0, -1.0)
    fitness = Fitness()

    invalid_values: tuple[float, ...] = (10.0,)

    with pytest.raises(ValueError) as exception_info:
        fitness.values = invalid_values

    assert "Mismatched dimensions" in str(exception_info.value)


@pytest.mark.usefixtures("configure_weights")
def test_setattr_allows_deap_to_handle_invalid_types() -> None:
    Fitness.weights = (1.0, 1.0)
    fitness = Fitness()

    with pytest.raises(TypeError):
        setattr(fitness, "values", 42.0)


@pytest.mark.usefixtures("configure_weights")
def test_setattr_ignores_other_attribute_names() -> None:
    Fitness.weights = (1.0, 1.0)
    fitness = Fitness()

    setattr(fitness, "some_other_attribute", (1.0, 2.0, 3.0))
    assert getattr(fitness, "some_other_attribute") == (1.0, 2.0, 3.0)
