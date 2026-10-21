import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    fitness_class_cache as cache_module,
)
from asian_asteroid_destroyers.genetic_algorithm.fitness import Fitness
from asian_asteroid_destroyers.genetic_algorithm.fitness_class_cache import (
    clear_fitness_cache,
    get_fitness_class,
)


@pytest.fixture(autouse=True)
def clear_module_attributes(request: pytest.FixtureRequest) -> None:
    request.addfinalizer(clear_fitness_cache)


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

    assert hasattr(cache_module, class_name)
    assert getattr(cache_module, class_name) is fitness_class
    assert fitness_class.__module__ == cache_module.__name__


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

    assert hasattr(cache_module, class_name)

    cache_module.clear_fitness_cache()
    assert not hasattr(cache_module, class_name)

    new_class_instance = get_fitness_class(weights)
    assert new_class_instance is not fitness_class
