import sys

import pytest

from asian_asteroid_destroyers.evokit import (
    Fitness,
    Individual,
    clear_fitness_cache,
    clear_individual_cache,
    get_individual_class,
    is_dynamic_individual_name,
    reconstruct_individual_weights,
)


@pytest.fixture
def configure_weights(request: pytest.FixtureRequest) -> None:
    def reset_weights() -> None:
        Fitness.weights = ()

    request.addfinalizer(reset_weights)


@pytest.fixture(autouse=True)
def clear_module_attributes(request: pytest.FixtureRequest) -> None:
    clear_individual_cache()
    clear_fitness_cache()

    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


def test_individual_initializes_empty() -> None:
    individual = Individual()
    assert len(individual) == 0
    assert isinstance(individual.fitness, Fitness)
    assert not individual.fitness.values


def test_individual_initializes_with_iterable() -> None:
    data = [1.5, 2.5, 3.5]
    individual = Individual(data)
    assert list(individual) == data
    assert isinstance(individual.fitness, Fitness)


def test_individual_initializes_with_tuple() -> None:
    data = (10.0, 20.0)
    individual = Individual(data)
    assert list(individual) == [10.0, 20.0]


@pytest.mark.usefixtures("configure_weights")
def test_individual_keeps_subclass_fitness() -> None:
    Fitness.weights = (1.0,)

    class CustomIndividual(Individual):
        fitness: Fitness

        def __init__(self) -> None:
            self.fitness = Fitness((1.0,))
            super().__init__([1.0])

    individual = CustomIndividual()
    assert hasattr(individual.fitness, "values")
    assert individual.fitness.values == (1.0,)


@pytest.mark.usefixtures("configure_weights")
def test_individual_fitness_dimension_validation() -> None:
    Fitness.weights = (-1.0, 1.0)
    individual = Individual([0.5, 0.5])

    individual.fitness.values = (10.0, 20.0)
    assert individual.fitness.values == (10.0, 20.0)

    with pytest.raises(ValueError, match="Mismatched dimensions"):
        individual.fitness.values = (10.0,)


def test_individual_list_mutations() -> None:
    individual = Individual([1.0, 2.0])
    individual.append(3.0)
    assert list(individual) == [1.0, 2.0, 3.0]


@pytest.mark.usefixtures("configure_weights")
def test_multi_objective_fitness_equality() -> None:
    Fitness.weights = (-1.0, 1.0, -1.0)
    individual = Individual([1.0, 2.0, 3.0])
    individual.fitness.values = (0.5, 10.0, -2.3)

    assert individual.fitness.values == (0.5, 10.0, -2.3)


def test_get_individual_class_creates_valid_subclass() -> None:
    weights: tuple[float, ...] = (1.0, -2.5)
    individual_class = get_individual_class(weights)

    assert issubclass(individual_class, Individual)
    assert individual_class.__name__ == "DynamicIndividual_1_0_neg2_5"


def test_get_individual_class_injects_into_sys_modules() -> None:
    weights: tuple[float, ...] = (1.0,)
    individual_class = get_individual_class(weights)
    class_name = individual_class.__name__
    main_module = sys.modules["__main__"]

    assert hasattr(main_module, class_name)
    assert getattr(main_module, class_name) is individual_class
    assert individual_class.__module__ == "__main__"


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
    main_module = sys.modules["__main__"]

    assert hasattr(main_module, class_name)

    clear_individual_cache()
    assert not hasattr(main_module, class_name)

    new_class_instance = get_individual_class(weights)
    assert new_class_instance is not individual_class


def test_is_dynamic_individual_name_matches_prefix() -> None:
    assert is_dynamic_individual_name("DynamicIndividual_1_0")
    assert not is_dynamic_individual_name("DynamicFitness_1_0")
    assert not is_dynamic_individual_name("SomeOtherClassName")


def test_reconstruct_individual_weights_extracts_tuple() -> None:
    expected = (-0.5, 100.25)
    class_name = "DynamicIndividual_neg0_5_100_25"

    result = reconstruct_individual_weights(class_name)
    assert result == expected
