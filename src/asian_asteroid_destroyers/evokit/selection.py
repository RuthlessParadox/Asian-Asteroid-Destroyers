from typing import Callable, cast

import numpy as np
from deap import tools

from .types import FloatMatrix, Population, SelectionOperator


def sel_tournament(
    tournament_size: int,
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_tournament = cast(
        Callable[[Population, int, int, str], Population],
        tools.selTournament,
    )

    def sel_tournament_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_tournament(
            population,
            k,
            tournament_size,
            fitness_attribute,
        )

    return sel_tournament_wrapper


def sel_roulette(
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_roulette = cast(
        Callable[[Population, int, str], Population],
        tools.selRoulette,
    )

    def sel_roulette_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_roulette(population, k, fitness_attribute)

    return sel_roulette_wrapper


def sel_nsga_2(
    algorithm: str = "standard",
) -> SelectionOperator:
    typed_sel_nsga_2 = cast(
        Callable[[Population, int, str], Population],
        tools.selNSGA2,
    )

    def sel_nsga_2_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_nsga_2(population, k, algorithm)

    return sel_nsga_2_wrapper


def sel_nsga_3(
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

    def sel_nsga_3_wrapper(
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

    return sel_nsga_3_wrapper


def sel_spea_2() -> SelectionOperator:
    typed_sel_spea_2 = cast(
        Callable[[Population, int], Population],
        tools.selSPEA2,
    )

    def sel_spea_2_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_spea_2(population, k)

    return sel_spea_2_wrapper


def sel_random() -> SelectionOperator:
    typed_sel_random = cast(
        Callable[[Population, int], Population],
        tools.selRandom,
    )

    def sel_random_wrapper(population: Population, k: int) -> Population:
        return typed_sel_random(population, k)

    return sel_random_wrapper


def sel_best(fitness_attribute: str = "fitness") -> SelectionOperator:
    typed_sel_best = cast(
        Callable[[Population, int, str], Population],
        tools.selBest,
    )

    def sel_best_wrapper(population: Population, k: int) -> Population:
        return typed_sel_best(population, k, fitness_attribute)

    return sel_best_wrapper


def sel_worst(fitness_attribute: str = "fitness") -> SelectionOperator:
    typed_sel_worst = cast(
        Callable[[Population, int, str], Population],
        tools.selWorst,
    )

    def sel_worst_wrapper(population: Population, k: int) -> Population:
        return typed_sel_worst(population, k, fitness_attribute)

    return sel_worst_wrapper


def sel_double_tournament(
    fitness_size: int,
    parsimony_size: float,
    fitness_first: bool,
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_double_tournament = cast(
        Callable[[Population, int, int, float, bool, str], Population],
        tools.selDoubleTournament,
    )

    def sel_double_tournament_wrapper(
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

    return sel_double_tournament_wrapper


def sel_stochastic_universal_sampling(
    fitness_attribute: str = "fitness",
) -> SelectionOperator:
    typed_sel_stochastic_universal_sampling = cast(
        Callable[[Population, int, str], Population],
        tools.selStochasticUniversalSampling,
    )

    def sel_stochastic_universal_sampling_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_stochastic_universal_sampling(
            population,
            k,
            fitness_attribute,
        )

    return sel_stochastic_universal_sampling_wrapper


def sel_tournament_dcd() -> SelectionOperator:
    typed_sel_tournament_dcd = cast(
        Callable[[Population, int], Population],
        tools.selTournamentDCD,
    )

    def sel_tournament_dcd_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_tournament_dcd(population, k)

    return sel_tournament_dcd_wrapper


def sel_lexicase() -> SelectionOperator:
    typed_sel_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selLexicase,
    )

    def sel_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_lexicase(population, k)

    return sel_lexicase_wrapper


def sel_epsilon_lexicase(epsilon: float) -> SelectionOperator:
    typed_sel_epsilon_lexicase = cast(
        Callable[[Population, int, float], Population],
        tools.selEpsilonLexicase,
    )

    def sel_epsilon_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_epsilon_lexicase(population, k, epsilon)

    return sel_epsilon_lexicase_wrapper


def sel_automatic_epsilon_lexicase() -> SelectionOperator:
    typed_sel_automatic_epsilon_lexicase = cast(
        Callable[[Population, int], Population],
        tools.selAutomaticEpsilonLexicase,
    )

    def sel_automatic_epsilon_lexicase_wrapper(
        population: Population,
        k: int,
    ) -> Population:
        return typed_sel_automatic_epsilon_lexicase(population, k)

    return sel_automatic_epsilon_lexicase_wrapper
