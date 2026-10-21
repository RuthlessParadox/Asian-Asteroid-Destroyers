import pytest

from asian_asteroid_destroyers.genetic_algorithm.fitness import Fitness
from asian_asteroid_destroyers.genetic_algorithm.individual import Individual


@pytest.fixture
def configure_weights(request: pytest.FixtureRequest) -> None:
    def reset_weights() -> None:
        Fitness.weights = ()

    request.addfinalizer(reset_weights)


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
    ind = Individual([1.0, 2.0, 3.0])
    ind.fitness.values = (0.5, 10.0, -2.3)

    assert ind.fitness.values == (0.5, 10.0, -2.3)
