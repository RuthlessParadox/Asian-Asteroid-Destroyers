from collections.abc import Sequence
from typing import Callable, NamedTuple, TypeVar, cast

import numpy as np
from deap import tools

from asian_asteroid_destroyers.genetic_algorithm.components import (
    Individual,
    Population,
    SelectionOperator,
)

_Shape1D = TypeVar("_Shape1D", bound=tuple[int])
_Shape2D = TypeVar("_Shape2D", bound=tuple[int, int])

FloatVector = np.ndarray[_Shape1D, np.dtype[np.float64]]
FloatMatrix = np.ndarray[_Shape2D, np.dtype[np.float64]]


def crossover_one_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOnePoint,
    )
    return typed_cx_one_point(individual_1, individual_2)


def crossover_two_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxTwoPoint,
    )
    return typed_cx_two_point(individual_1, individual_2)


def crossover_uniform(
    individual_1: Individual,
    individual_2: Individual,
    independent_probability: float,
) -> tuple[Individual, Individual]:
    typed_cx_uniform = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniform,
    )
    return typed_cx_uniform(
        individual_1,
        individual_2,
        independent_probability,
    )


def crossover_partially_matched(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_partially_matched = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxPartialyMatched,
    )
    return typed_cx_partially_matched(individual_1, individual_2)


def crossover_uniform_partially_matched(
    individual_1: Individual,
    individual_2: Individual,
    independent_probability: float,
) -> tuple[Individual, Individual]:
    typed_cx_uniform_partially_matched = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniformPartialyMatched,
    )
    return typed_cx_uniform_partially_matched(
        individual_1,
        individual_2,
        independent_probability,
    )


def crossover_ordered(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_ordered = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOrdered,
    )
    return typed_cx_ordered(individual_1, individual_2)


def crossover_blend(
    individual_1: Individual,
    individual_2: Individual,
    alpha: float,
) -> tuple[Individual, Individual]:
    typed_cx_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxBlend,
    )
    return typed_cx_blend(individual_1, individual_2, alpha)


def crossover_evolution_strategy_blend(
    individual_1: Individual,
    individual_2: Individual,
    alpha: float,
) -> tuple[Individual, Individual]:
    typed_cx_evolution_strategy_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxESBlend,
    )
    return typed_cx_evolution_strategy_blend(individual_1, individual_2, alpha)


def crossover_evolution_strategy_two_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_evolution_strategy_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxESTwoPoint,
    )
    return typed_cx_evolution_strategy_two_point(individual_1, individual_2)


def crossover_simulated_binary(
    individual_1: Individual,
    individual_2: Individual,
    eta: float,
) -> tuple[Individual, Individual]:
    typed_cx_simulated_binary = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxSimulatedBinary,
    )
    return typed_cx_simulated_binary(individual_1, individual_2, eta)


def crossover_simulated_binary_bounded(
    individual_1: Individual,
    individual_2: Individual,
    eta: float,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
) -> tuple[Individual, Individual]:
    typed_cx_simulated_binary_bounded = cast(
        Callable[
            [
                Individual,
                Individual,
                float,
                float | Sequence[float],
                float | Sequence[float],
            ],
            tuple[Individual, Individual],
        ],
        tools.cxSimulatedBinaryBounded,
    )
    return typed_cx_simulated_binary_bounded(
        individual_1, individual_2, eta, lower_bound, upper_bound
    )


def crossover_messy_one_point(
    individual_1: Individual,
    individual_2: Individual,
) -> tuple[Individual, Individual]:
    typed_cx_messy_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxMessyOnePoint,
    )
    return typed_cx_messy_one_point(individual_1, individual_2)


def mutation_gaussian(
    individual: Individual,
    mu: float | Sequence[float],
    sigma: float | Sequence[float],
    independent_probability: float,
) -> tuple[Individual]:
    typed_mut_gaussian = cast(
        Callable[
            [
                Individual,
                float | Sequence[float],
                float | Sequence[float],
                float,
            ],
            tuple[Individual],
        ],
        tools.mutGaussian,
    )
    return typed_mut_gaussian(individual, mu, sigma, independent_probability)


def mutation_shuffle_indices(
    individual: Individual,
    independent_probability: float,
) -> tuple[Individual]:
    typed_mut_shuffle_indices = cast(
        Callable[[Individual, float], tuple[Individual]],
        tools.mutShuffleIndexes,
    )
    return typed_mut_shuffle_indices(individual, independent_probability)


def mutation_flip_bit(
    individual: Individual,
    independent_probability: float,
) -> tuple[Individual]:
    typed_mut_flip_bit = cast(
        Callable[[Individual, float], tuple[Individual]],
        tools.mutFlipBit,
    )
    return typed_mut_flip_bit(individual, independent_probability)


def mutation_uniform_int(
    individual: Individual,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    independent_probability: float,
) -> tuple[Individual]:
    typed_mut_uniform_int = cast(
        Callable[
            [
                Individual,
                float | Sequence[float],
                float | Sequence[float],
                float,
            ],
            tuple[Individual],
        ],
        tools.mutUniformInt,
    )
    return typed_mut_uniform_int(
        individual, lower_bound, upper_bound, independent_probability
    )


def mutation_polynomial_bounded(
    individual: Individual,
    eta: float,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    independent_probability: float,
) -> tuple[Individual]:
    typed_mut_polynomial_bounded = cast(
        Callable[
            [
                Individual,
                float,
                float | Sequence[float],
                float | Sequence[float],
                float,
            ],
            tuple[Individual],
        ],
        tools.mutPolynomialBounded,
    )
    return typed_mut_polynomial_bounded(
        individual,
        eta,
        lower_bound,
        upper_bound,
        independent_probability,
    )


def mutation_evolution_strategy_log_normal(
    individual: Individual,
    c: float,
    independent_probability: float,
) -> tuple[Individual]:
    typed_mut_evolution_strategy_log_normal = cast(
        Callable[[Individual, float, float], tuple[Individual]],
        tools.mutESLogNormal,
    )
    return typed_mut_evolution_strategy_log_normal(
        individual,
        c,
        independent_probability,
    )


def selection_tournament(
    population: Population,
    k: int,
    tournament_size: int,
) -> Population:
    typed_sel_tournament = cast(
        Callable[[Population, int, int], Population],
        tools.selTournament,
    )
    return typed_sel_tournament(
        population,
        k,
        tournament_size,
    )


def selection_roulette(
    population: Population,
    k: int,
) -> Population:
    typed_sel_roulette = cast(
        Callable[[Population, int], Population],
        tools.selRoulette,
    )
    return typed_sel_roulette(population, k)


def selection_non_dominated_sorting_genetic_algorithm_2(
    population: Population,
    k: int,
    algorithm: str = "standard",
) -> Population:
    typed_sel_nsga_2 = cast(
        Callable[[Population, int, str], Population],
        tools.selNSGA2,
    )
    return typed_sel_nsga_2(population, k, algorithm)


class NonDominatedSortingGeneticAlgorithm3Memory(NamedTuple):
    best_point: FloatVector[tuple[int]] | None
    worst_point: FloatVector[tuple[int]] | None
    extreme_points: FloatMatrix[tuple[int, int]] | None


def selection_non_dominated_sorting_genetic_algorithm_3(
    population: Population,
    k: int,
    reference_points: FloatMatrix[tuple[int, int]],
    algorithm: str = "log",
    best_point: FloatVector[tuple[int]] | None = None,
    worst_point: FloatVector[tuple[int]] | None = None,
    extreme_points: FloatMatrix[tuple[int, int]] | None = None,
    return_memory: bool = False,
) -> (
    Population | tuple[Population, NonDominatedSortingGeneticAlgorithm3Memory]
):
    typed_sel_nsga_3 = cast(
        Callable[
            [
                Population,
                int,
                FloatMatrix[tuple[int, int]],
                str,
                FloatVector[tuple[int]] | None,
                FloatVector[tuple[int]] | None,
                FloatMatrix[tuple[int, int]] | None,
                bool,
            ],
            Population
            | tuple[Population, NonDominatedSortingGeneticAlgorithm3Memory],
        ],
        tools.selNSGA3,
    )
    return typed_sel_nsga_3(
        population,
        k,
        reference_points,
        algorithm,
        best_point,
        worst_point,
        extreme_points,
        return_memory,
    )


class _NSGA3Proxy:
    def __init__(self, ref_points: FloatMatrix[tuple[int, int]]) -> None:
        self._selector: tuple[object, ...] = (
            tools.selNSGA3WithMemory(ref_points.tolist()),
        )

    def __call__(
        self,
        individuals: Population,
        k: int,
        non_dominated_values: FloatMatrix[tuple[int, int]],
    ) -> Population:
        raw_function = self._selector[0]
        operator = cast(
            Callable[[Population, int, list[list[float]]], Population],
            raw_function,
        )
        return operator(individuals, k, non_dominated_values.tolist())


def get_selection_non_dominated_sorting_genetic_algorithm_3_with_memory(
    reference_points: FloatMatrix[tuple[int, int]],
) -> SelectionOperator:
    nsga_3_instance = _NSGA3Proxy(reference_points)

    def nsga3_selector_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        raw_list = [
            list(individual.fitness.values) for individual in population
        ]
        non_dominated_values = cast(
            FloatMatrix[tuple[int, int]],
            np.array(raw_list, dtype=np.float64),
        )
        return nsga_3_instance(population, k, non_dominated_values)

    return nsga3_selector_wrapper


def selection_strength_pareto_evolutionary_algorithm_2(
    population: Population,
    k: int,
) -> Population:
    typed_sel_spea_2 = cast(
        Callable[[Population, int], Population],
        tools.selSPEA2,
    )
    return typed_sel_spea_2(population, k)


def selection_random(
    population: Population,
    k: int,
) -> Population:
    typed_sel_random = cast(
        Callable[[Population, int], Population],
        tools.selRandom,
    )
    return typed_sel_random(population, k)


def selection_best(
    population: Population,
    k: int,
) -> Population:
    typed_sel_best = cast(
        Callable[[Population, int], Population],
        tools.selBest,
    )
    return typed_sel_best(population, k)


def selection_worst(
    population: Population,
    k: int,
) -> Population:
    typed_sel_worst = cast(
        Callable[[Population, int], Population],
        tools.selWorst,
    )
    return typed_sel_worst(population, k)


def selection_double_tournament(
    population: Population,
    k: int,
    fitness_size: int,
    parsimony_size: float,
    fitness_first: bool,
) -> Population:
    typed_sel_double_tournament = cast(
        Callable[[Population, int, int, float, bool], Population],
        tools.selDoubleTournament,
    )
    return typed_sel_double_tournament(
        population,
        k,
        fitness_size,
        parsimony_size,
        fitness_first,
    )


def selection_stochastic_universal_sampling(
    population: Population,
    k: int,
) -> Population:
    typed_sel_stochastic_universal_sampling = cast(
        Callable[[Population, int], Population],
        tools.selStochasticUniversalSampling,
    )
    return typed_sel_stochastic_universal_sampling(population, k)


def selection_tournament_dominance_crowding_distance(
    population: Population,
    k: int,
) -> Population:
    typed_sel_tournament_dcd = cast(
        Callable[[Population, int], Population],
        tools.selTournamentDCD,
    )
    return typed_sel_tournament_dcd(population, k)


def selection_lexicase(
    population: Population,
    k: int,
) -> Population:
    typed_sel_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selLexicase,
    )
    return typed_sel_lexicase(population, k)


def selection_epsilon_lexicase(
    population: Population,
    k: int,
    epsilon: float,
) -> Population:
    typed_sel_epsilon_lexicase = cast(
        Callable[[Population, int, float], Population],
        tools.selEpsilonLexicase,
    )
    return typed_sel_epsilon_lexicase(population, k, epsilon)


def selection_automatic_epsilon_lexicase(
    population: Population,
    k: int,
) -> Population:
    typed_sel_automatic_epsilon_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selAutomaticEpsilonLexicase,
    )
    return typed_sel_automatic_epsilon_lexicase(population, k)
