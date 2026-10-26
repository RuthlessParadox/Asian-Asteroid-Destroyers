import pytest
from deap import base

from asian_asteroid_destroyers.genetic_algorithm import (
    Fitness,
    clear_fitness_cache,
    fitness as fitness_module,
    get_fitness_class,
)


@pytest.fixture
def configure_weights(request: pytest.FixtureRequest) -> None:
    def reset_weights() -> None:
        Fitness.weights = ()

    request.addfinalizer(reset_weights)


@pytest.fixture(autouse=True)
def clear_module_attributes(request: pytest.FixtureRequest) -> None:
    request.addfinalizer(clear_fitness_cache)


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


def test_get_fitness_class_creates_valid_subclass() -> None:
    weights: tuple[float, ...] = (1.0, -2.5)
    fitness_class = get_fitness_class(weights)

    assert issubclass(fitness_class, Fitness)
    assert fitness_class.weights == weights
    assert fitness_class.__name__ == "DynamicFitness_1_0_neg2_5"


def test_get_fitness_class_injects_into_sys_modules() -> None:
    weights: tuple[float, ...] = (1.0,)
    fitness_class = get_fitness_class(weights)
    class_name = fitness_class.__name__

    assert hasattr(fitness_module, class_name)
    assert getattr(fitness_module, class_name) is fitness_class
    assert fitness_class.__module__ == fitness_module.__name__


def test_get_fitness_class_returns_identical_class_for_same_weights() -> None:
    weights: tuple[float, ...] = (0.5, 0.5)
    first_class = get_fitness_class(weights)
    second_class = get_fitness_class(weights)

    assert first_class is second_class


def test_dynamic_fitness_instance_validation_success() -> None:
    weights: tuple[float, ...] = (1.0, 1.0)
    fitness_class = get_fitness_class(weights)
    instance = fitness_class()

    instance.values = (10.0, 20.0)
    assert instance.values == (10.0, 20.0)


def test_dynamic_fitness_instance_validation_failure() -> None:
    weights: tuple[float, ...] = (1.0, 1.0)
    fitness_class = get_fitness_class(weights)
    instance = fitness_class()

    with pytest.raises(ValueError, match="Mismatched dimensions"):
        instance.values = (10.0,)


def test_get_fitness_class_handles_empty_weight_tuples() -> None:
    weights: tuple[float, ...] = ()
    fitness_class = get_fitness_class(weights)

    assert issubclass(fitness_class, Fitness)
    assert fitness_class.weights == ()
    assert fitness_class.__name__ == "DynamicFitness_"

    instance = fitness_class()
    instance.values = ()
    assert instance.values == ()


def test_get_fitness_class_handles_complex_floating_point_names() -> None:
    weights: tuple[float, ...] = (-0.0001, 123.456, -7.0)
    fitness_class = get_fitness_class(weights)

    expected_name = "DynamicFitness_neg0_0001_123_456_neg7_0"
    assert fitness_class.__name__ == expected_name
    assert fitness_class.weights == weights


def test_get_fitness_class_rejects_extreme_and_infinite_values() -> None:
    inf_value = float("inf")
    neginf_value = float("-inf")
    nan_value = float("nan")

    with pytest.raises(
        ValueError,
        match="Fitness weights must be finite numbers",
    ):
        _ = get_fitness_class((inf_value, 1.0))

    with pytest.raises(
        ValueError,
        match="Fitness weights must be finite numbers",
    ):
        _ = get_fitness_class((1.0, neginf_value))

    with pytest.raises(
        ValueError,
        match="Fitness weights must be finite numbers",
    ):
        _ = get_fitness_class((1.0, nan_value))


def test_clear_fitness_cache_removes_classes_and_empties_dict() -> None:
    weights: tuple[float, ...] = (4.2,)
    fitness_class = get_fitness_class(weights)
    class_name = fitness_class.__name__

    assert hasattr(fitness_module, class_name)

    fitness_module.clear_fitness_cache()
    assert not hasattr(fitness_module, class_name)

    new_class_instance = get_fitness_class(weights)
    assert new_class_instance is not fitness_class
