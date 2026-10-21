import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    individual_class_cache as cache_module,
)
from asian_asteroid_destroyers.genetic_algorithm.fitness_class_cache import (
    clear_fitness_cache,
)
from asian_asteroid_destroyers.genetic_algorithm.individual import (
    Individual,
)
from asian_asteroid_destroyers.genetic_algorithm.individual_class_cache import (
    clear_individual_cache,
    get_individual_class,
)


@pytest.fixture(autouse=True)
def clear_module_attributes(request: pytest.FixtureRequest) -> None:
    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


def test_get_individual_class_creates_valid_subclass() -> None:
    weights: tuple[float, ...] = (1.0, -2.5)
    individual_class = get_individual_class(weights)

    assert issubclass(individual_class, Individual)
    assert individual_class.__name__ == "DynamicIndividual_1_0_neg2_5"


def test_get_individual_class_injects_into_sys_modules() -> None:
    weights: tuple[float, ...] = (1.0,)
    individual_class = get_individual_class(weights)
    class_name = individual_class.__name__

    assert hasattr(cache_module, class_name)
    assert getattr(cache_module, class_name) is individual_class
    assert individual_class.__module__ == cache_module.__name__


def test_get_individual_class_returns_identical_class_for_same_weights() -> (
    None
):
    weights: tuple[float, ...] = (0.5, 0.5)
    first_class = get_individual_class(weights)
    second_class = get_individual_class(weights)

    assert first_class is second_class


def test_get_individual_class_handles_empty_weight_tuples() -> None:
    weights: tuple[float, ...] = ()
    individual_class = get_individual_class(weights)

    assert issubclass(individual_class, Individual)
    assert individual_class.__name__ == "DynamicIndividual_"

    instance = individual_class([5.5, 6.6])
    assert instance == [5.5, 6.6]
    assert instance.fitness.weights == ()


def test_get_individual_class_handles_complex_floating_point_names() -> None:
    weights: tuple[float, ...] = (-0.0001, 123.456, -7.0)
    individual_class = get_individual_class(weights)

    expected_name = "DynamicIndividual_neg0_0001_123_456_neg7_0"
    assert individual_class.__name__ == expected_name


def test_dynamic_individual_initializes_with_provided_iterable() -> None:
    weights: tuple[float, ...] = (1.0, -1.0)
    individual_class = get_individual_class(weights)
    initial_elements = [1.2, 3.4, 5.6]

    individual = individual_class(initial_elements)

    assert individual == initial_elements
    assert isinstance(individual, list)


def test_dynamic_individual_initializes_empty_by_default() -> None:
    weights: tuple[float, ...] = (1.0, -1.0)
    individual_class = get_individual_class(weights)

    individual = individual_class()

    assert individual == []


def test_dynamic_individual_contains_validated_fitness_instance() -> None:
    weights: tuple[float, ...] = (1.0, 1.0)
    individual_class = get_individual_class(weights)
    individual = individual_class()

    assert individual.fitness.weights == weights

    individual.fitness.values = (10.0, 20.0)
    assert individual.fitness.values == (10.0, 20.0)

    with pytest.raises(ValueError, match="Mismatched dimensions"):
        individual.fitness.values = (5.0,)


def test_clear_individual_cache_removes_classes_and_empties_dict() -> None:
    weights: tuple[float, ...] = (4.2,)
    individual_class = get_individual_class(weights)
    class_name = individual_class.__name__

    assert hasattr(cache_module, class_name)

    cache_module.clear_individual_cache()
    assert not hasattr(cache_module, class_name)

    new_class_instance = get_individual_class(weights)
    assert new_class_instance is not individual_class
