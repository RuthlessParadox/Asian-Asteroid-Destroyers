from typing import Callable, cast

import numpy as np
from deap import tools

from .types import FloatMatrix, Population, SelectionOperator


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
