import copy
from collections.abc import Sequence
from typing import Callable, cast

import numpy as np
from deap import tools

from .fitness_class_cache import get_fitness_class
from .individual import Individual
from .types import (
    CrossoverOperator,
    FloatMatrix,
    MutationOperator,
    Population,
    SelectionOperator,
)


def reweight_population(
    population: Population,
    new_weights: tuple[float, ...],
) -> None:
    new_fitness_type = get_fitness_class(new_weights)
    empty_prototype = new_fitness_type()

    for individual in population:
        fitness = individual.fitness
        raw_values: tuple[float, ...] | None = None

        if getattr(fitness, "valid", False):
            try:
                raw_values = fitness.values
            except AttributeError:
                pass

        if raw_values is not None:
            new_fitness = new_fitness_type()
            new_fitness.values = raw_values
            individual.fitness = new_fitness
        else:
            individual.fitness = copy.copy(empty_prototype)


def crossover_one_point() -> CrossoverOperator:
    typed_cx_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOnePoint,
    )

    def crossover_one_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_one_point(individual_1, individual_2)

    return crossover_one_point_wrapper


def crossover_two_point() -> CrossoverOperator:
    typed_cx_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxTwoPoint,
    )

    def crossover_two_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_two_point(individual_1, individual_2)

    return crossover_two_point_wrapper


def crossover_uniform(independent_probability: float) -> CrossoverOperator:
    typed_cx_uniform = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniform,
    )

    def crossover_uniform_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_uniform(
            individual_1,
            individual_2,
            independent_probability,
        )

    return crossover_uniform_wrapper


def crossover_partially_matched() -> CrossoverOperator:
    typed_cx_partially_matched = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxPartialyMatched,
    )

    def crossover_partially_matched_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_partially_matched(individual_1, individual_2)

    return crossover_partially_matched_wrapper


def crossover_uniform_partially_matched(
    independent_probability: float,
) -> CrossoverOperator:
    typed_cx_uniform_partially_matched = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxUniformPartialyMatched,
    )

    def crossover_uniform_partially_matched_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_uniform_partially_matched(
            individual_1,
            individual_2,
            independent_probability,
        )

    return crossover_uniform_partially_matched_wrapper


def crossover_ordered() -> CrossoverOperator:
    typed_cx_ordered = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxOrdered,
    )

    def crossover_ordered_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_ordered(individual_1, individual_2)

    return crossover_ordered_wrapper


def crossover_blend(alpha: float) -> CrossoverOperator:
    typed_cx_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxBlend,
    )

    def crossover_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_blend(individual_1, individual_2, alpha)

    return crossover_blend_wrapper


def crossover_evolution_strategy_blend(alpha: float) -> CrossoverOperator:
    typed_cx_es_blend = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxESBlend,
    )

    def crossover_evolution_strategy_blend_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_es_blend(individual_1, individual_2, alpha)

    return crossover_evolution_strategy_blend_wrapper


def crossover_evolution_strategy_two_point() -> CrossoverOperator:
    typed_cx_es_two_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxESTwoPoint,
    )

    def crossover_evolution_strategy_two_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_es_two_point(individual_1, individual_2)

    return crossover_evolution_strategy_two_point_wrapper


def crossover_simulated_binary(eta: float) -> CrossoverOperator:
    typed_cx_simulated_binary = cast(
        Callable[
            [Individual, Individual, float],
            tuple[Individual, Individual],
        ],
        tools.cxSimulatedBinary,
    )

    def crossover_simulated_binary_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_simulated_binary(individual_1, individual_2, eta)

    return crossover_simulated_binary_wrapper


def crossover_simulated_binary_bounded(
    eta: float,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
) -> CrossoverOperator:
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

    def crossover_simulated_binary_bounded_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_simulated_binary_bounded(
            individual_1,
            individual_2,
            eta,
            lower_bound,
            upper_bound,
        )

    return crossover_simulated_binary_bounded_wrapper


def crossover_messy_one_point() -> CrossoverOperator:
    typed_cx_messy_one_point = cast(
        Callable[[Individual, Individual], tuple[Individual, Individual]],
        tools.cxMessyOnePoint,
    )

    def crossover_messy_one_point_wrapper(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return typed_cx_messy_one_point(individual_1, individual_2)

    return crossover_messy_one_point_wrapper


def mutation_gaussian(
    mu: float | Sequence[float],
    sigma: float | Sequence[float],
    independent_probability: float,
) -> MutationOperator:
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

    def mutation_gaussian_wrapper(individual: Individual) -> tuple[Individual]:
        return typed_mut_gaussian(
            individual,
            mu,
            sigma,
            independent_probability,
        )

    return mutation_gaussian_wrapper


def mutation_shuffle_indices(
    independent_probability: float,
) -> MutationOperator:
    typed_mut_shuffle_indices = cast(
        Callable[[Individual, float], tuple[Individual]],
        tools.mutShuffleIndexes,
    )

    def mutation_shuffle_indices_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_shuffle_indices(individual, independent_probability)

    return mutation_shuffle_indices_wrapper


def mutation_flip_bit(independent_probability: float) -> MutationOperator:
    typed_mut_flip_bit = cast(
        Callable[[Individual, float], tuple[Individual]],
        tools.mutFlipBit,
    )

    def mutation_flip_bit_wrapper(individual: Individual) -> tuple[Individual]:
        return typed_mut_flip_bit(individual, independent_probability)

    return mutation_flip_bit_wrapper


def mutation_uniform_int(
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    independent_probability: float,
) -> MutationOperator:
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

    def mutation_uniform_int_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_uniform_int(
            individual,
            lower_bound,
            upper_bound,
            independent_probability,
        )

    return mutation_uniform_int_wrapper


def mutation_polynomial_bounded(
    eta: float,
    lower_bound: float | Sequence[float],
    upper_bound: float | Sequence[float],
    independent_probability: float,
) -> MutationOperator:
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

    def mutation_polynomial_bounded_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_polynomial_bounded(
            individual,
            eta,
            lower_bound,
            upper_bound,
            independent_probability,
        )

    return mutation_polynomial_bounded_wrapper


def mutation_evolution_strategy_log_normal(
    c: float,
    independent_probability: float,
) -> MutationOperator:
    typed_mut_es_log_normal = cast(
        Callable[[Individual, float, float], tuple[Individual]],
        tools.mutESLogNormal,
    )

    def mutation_evolution_strategy_log_normal_wrapper(
        individual: Individual,
    ) -> tuple[Individual]:
        return typed_mut_es_log_normal(individual, c, independent_probability)

    return mutation_evolution_strategy_log_normal_wrapper


def selection_tournament(
    tournament_size: int,
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_tournament = cast(
        Callable[[Population, int, int, str], Population],
        tools.selTournament,
    )

    def selection_tournament_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_tournament(
            population,
            k,
            tournament_size,
            fitness_attribute,
        )

    return selection_tournament_wrapper


def selection_roulette(
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_roulette = cast(
        Callable[[Population, int, str], Population],
        tools.selRoulette,
    )

    def selection_roulette_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_roulette(population, k, fitness_attribute)

    return selection_roulette_wrapper


def selection_non_dominated_sorting_genetic_algorithm_2(
    algorithm: str = "standard",
) -> SelectionOperator:
    typed_sel_nsga_2 = cast(
        Callable[[Population, int, str], Population],
        tools.selNSGA2,
    )

    def selection_non_dominated_sorting_genetic_algorithm_2_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_nsga_2(population, k, algorithm)

    return selection_non_dominated_sorting_genetic_algorithm_2_wrapper


def selection_non_dominated_sorting_genetic_algorithm_3(
    reference_points: FloatMatrix[tuple[int, int]],
) -> SelectionOperator:
    class NSGA3Proxy:
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

    nsga_3_instance = NSGA3Proxy(reference_points)

    def selection_non_dominated_sorting_genetic_algorithm_3_wrapper(
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

    return selection_non_dominated_sorting_genetic_algorithm_3_wrapper


def selection_strength_pareto_evolutionary_algorithm_2() -> SelectionOperator:
    typed_sel_spea_2 = cast(
        Callable[[Population, int], Population],
        tools.selSPEA2,
    )

    def selection_strength_pareto_evolutionary_algorithm_2_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_spea_2(population, k)

    return selection_strength_pareto_evolutionary_algorithm_2_wrapper


def selection_random() -> SelectionOperator:
    typed_sel_random = cast(
        Callable[[Population, int], Population],
        tools.selRandom,
    )

    def selection_random_wrapper(population: Population, k: int) -> Population:
        return typed_sel_random(population, k)

    return selection_random_wrapper


def selection_best(fitness_attribute: str = "fitness") -> SelectionOperator:
    typed_sel_best = cast(
        Callable[[Population, int, str], Population],
        tools.selBest,
    )

    def selection_best_wrapper(population: Population, k: int) -> Population:
        return typed_sel_best(population, k, fitness_attribute)

    return selection_best_wrapper


def selection_worst(fitness_attribute: str = "fitness") -> SelectionOperator:
    typed_sel_worst = cast(
        Callable[[Population, int, str], Population],
        tools.selWorst,
    )

    def selection_worst_wrapper(population: Population, k: int) -> Population:
        return typed_sel_worst(population, k, fitness_attribute)

    return selection_worst_wrapper


def selection_double_tournament(
    fitness_size: int,
    parsimony_size: float,
    fitness_first: bool,
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_double_tournament = cast(
        Callable[[Population, int, int, float, bool, str], Population],
        tools.selDoubleTournament,
    )

    def selection_double_tournament_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_double_tournament(
            population,
            k,
            fitness_size,
            parsimony_size,
            fitness_first,
            fitness_attribute,
        )

    return selection_double_tournament_wrapper


def selection_stochastic_universal_sampling(
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_stochastic_universal_sampling = cast(
        Callable[[Population, int, str], Population],
        tools.selStochasticUniversalSampling,
    )

    def selection_stochastic_universal_sampling_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_stochastic_universal_sampling(
            population,
            k,
            fitness_attribute,
        )

    return selection_stochastic_universal_sampling_wrapper


def selection_tournament_dominance_crowding_distance() -> SelectionOperator:
    typed_sel_tournament_dcd = cast(
        Callable[[Population, int], Population],
        tools.selTournamentDCD,
    )

    def selection_tournament_dominance_crowding_distance_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_tournament_dcd(population, k)

    return selection_tournament_dominance_crowding_distance_wrapper


def selection_lexicase() -> SelectionOperator:
    typed_sel_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selLexicase,
    )

    def selection_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_lexicase(population, k)

    return selection_lexicase_wrapper


def selection_epsilon_lexicase(epsilon: float) -> SelectionOperator:
    typed_sel_epsilon_lexicase = cast(
        Callable[[Population, int, float], Population],
        tools.selEpsilonLexicase,
    )

    def selection_epsilon_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_epsilon_lexicase(population, k, epsilon)

    return selection_epsilon_lexicase_wrapper


def selection_automatic_epsilon_lexicase() -> SelectionOperator:
    typed_sel_automatic_epsilon_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selAutomaticEpsilonLexicase,
    )

    def selection_automatic_epsilon_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_automatic_epsilon_lexicase(population, k)

    return selection_automatic_epsilon_lexicase_wrapper
