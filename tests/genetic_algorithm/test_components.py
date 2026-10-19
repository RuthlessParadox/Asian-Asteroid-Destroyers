from typing import Callable, Protocol, cast

import pytest
from deap import algorithms, base

from asian_asteroid_destroyers.genetic_algorithm.components import (
    GeneticAlgorithmToolbox,
    Individual,
    Population,
    TypedFitness,
    build_toolbox,
    reweight_population,
)


@pytest.fixture
def sample_evaluate_single() -> Callable[[Individual], tuple[float]]:
    def _evaluate(individual: Individual) -> tuple[float]:
        return (float(sum(individual)),)

    return _evaluate


@pytest.fixture
def sample_evaluate_multi() -> Callable[[Individual], tuple[float, float]]:
    def _evaluate(individual: Individual) -> tuple[float, float]:
        return float(sum(individual)), float(len(individual))

    return _evaluate


def test_typed_fitness_initialization() -> None:
    fitness = TypedFitness(values=(1.0, 2.5))
    assert fitness.values == (1.0, 2.5)

    empty_fitness = TypedFitness()
    assert not hasattr(empty_fitness, "values")


def test_individual_initialization() -> None:
    data = [1.0, 2.0, 3.0]
    ind = Individual(iterable=data)
    assert list(ind) == data
    assert isinstance(ind.fitness, TypedFitness)

    empty_ind = Individual()
    assert list(empty_ind) == []
    assert isinstance(empty_ind.fitness, TypedFitness)


def test_toolbox_unregistered_stubs_raise_error() -> None:
    toolbox = GeneticAlgorithmToolbox()

    with pytest.raises(NotImplementedError) as exception_info:
        _ = toolbox.individual()
    assert "individual" in str(exception_info.value)

    with pytest.raises(NotImplementedError):
        _ = toolbox.mate(Individual([1.0]), Individual([2.0]))


def test_toolbox_register_and_unregister(
    sample_evaluate_single: Callable[[Individual], tuple[float]],
) -> None:
    toolbox = GeneticAlgorithmToolbox()

    toolbox.register("evaluate", sample_evaluate_single)
    assert toolbox.evaluate is sample_evaluate_single
    assert toolbox.evaluate(Individual()) == (1.0,)

    toolbox.unregister("evaluate")
    with pytest.raises(AttributeError):
        _ = toolbox.evaluate


def test_build_toolbox_single_objective(
    sample_evaluate_single: Callable[[Individual], tuple[float]],
) -> None:
    weights = (1.0,)
    gene_size = 5

    toolbox = build_toolbox(
        evaluation_function=sample_evaluate_single,
        gene_size=gene_size,
        weights=weights,
    )

    population = toolbox.population(10)
    assert len(population) == 10

    for individual in population:
        assert len(individual) == gene_size
        assert isinstance(individual, Individual)
        assert individual.fitness.weights == weights


def test_build_toolbox_multi_objective(
    sample_evaluate_multi: Callable[[Individual], tuple[float, float]],
) -> None:
    weights = (-1.0, 1.0)
    gene_size = 3

    toolbox = build_toolbox(
        evaluation_function=sample_evaluate_multi,
        gene_size=gene_size,
        weights=weights,
    )

    population = toolbox.population(5)
    assert len(population) == 5

    for individual in population:
        assert len(individual) == gene_size
        assert isinstance(individual, Individual)
        assert individual.fitness.weights == weights


def test_build_toolbox_with_custom_operators(
    sample_evaluate_single: Callable[[Individual], tuple[float]],
) -> None:
    def custom_mate(
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]:
        return individual2, individual1

    toolbox = build_toolbox(
        evaluation_function=sample_evaluate_single,
        gene_size=4,
        mate_operator=custom_mate,
    )

    test_individual1 = Individual([1.0, 1.0])
    test_individual2 = Individual([2.0, 2.0])
    result1, result2 = toolbox.mate(test_individual1, test_individual2)
    assert result1 == test_individual1
    assert result2 == test_individual2


def test_reweight_population_preserves_unevaluated_states(
    sample_evaluate_multi: Callable[[Individual], tuple[float, float]],
) -> None:
    toolbox = build_toolbox(
        evaluation_function=sample_evaluate_multi,
        gene_size=2,
        weights=(1.0, 1.0),
    )
    population = toolbox.population(2)

    assert not population[0].fitness.valid

    population[1].fitness.values = (10.0, 20.0)
    assert population[1].fitness.valid

    new_weights = (-1.0, -1.0)
    reweight_population(population, new_weights)

    assert not population[0].fitness.valid
    assert population[0].fitness.weights == new_weights

    assert population[1].fitness.valid
    assert population[1].fitness.values == (10.0, 20.0)
    assert population[1].fitness.weights == new_weights


class _EaSimpleProtocol(Protocol):
    def __call__(
        self,
        population: Population,
        toolbox: base.Toolbox,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: object = None,
        halloffame: object = None,
        verbose: bool = False,
    ) -> tuple[Population, object]: ...


def test_integration_with_ea_simple(
    sample_evaluate_single: Callable[[Individual], tuple[float]],
) -> None:
    gene_size = 5
    target_weights = (1.0,)

    toolbox = build_toolbox(
        evaluation_function=sample_evaluate_single,
        gene_size=gene_size,
        weights=target_weights,
    )

    population = toolbox.population(20)
    cxpb = 0.5
    mutpb = 0.2
    ngen = 3

    ea_simple_typed = cast(_EaSimpleProtocol, algorithms.eaSimple)

    final_population, logbook = ea_simple_typed(
        population=population,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        verbose=False,
    )

    assert len(final_population) == 20
    assert logbook is not None

    for individual in final_population:
        assert isinstance(individual, Individual)
        assert individual.fitness.valid
        assert isinstance(individual.fitness.values[0], float)


def test_pitfall_mismatched_evaluation_dimensions(
    sample_evaluate_single: Callable[[Individual], tuple[float]],
) -> None:
    toolbox = build_toolbox(
        evaluation_function=sample_evaluate_single,
        gene_size=3,
        weights=(1.0, -1.0),
    )

    individual = toolbox.individual()

    with pytest.raises(ValueError):
        individual.fitness.values = toolbox.evaluate(individual)
