import copy
import random
from collections.abc import Callable, Iterable
from typing import TypeVar, cast

from deap import base, tools

_T = TypeVar("_T")
_R = TypeVar("_R")


class TypedFitness(base.Fitness):
    values: tuple[float, ...]
    weights: tuple[float, ...] = (1.0,)

    def __init__(self, values: tuple[float, ...] | None = None) -> None:
        super().__init__()
        if values is not None:
            self.values = values


class Individual(list[float]):
    fitness: TypedFitness

    def __init__(self, iterable: Iterable[float] | None = None) -> None:
        if iterable is not None:
            super().__init__(iterable)
        else:
            super().__init__()
        self.fitness = TypedFitness()


Population = list[Individual]
GeneGenerator = Callable[[], float]
IndividualGenerator = Callable[[], Individual]
PopulationGenerator = Callable[[int], Population]
FitnessEvaluator = Callable[[Individual], tuple[float, ...]]
CrossoverOperator = Callable[
    [Individual, Individual], tuple[Individual, Individual]
]
MutationOperator = Callable[[Individual], tuple[Individual]]
SelectionOperator = Callable[[Population, int], Population]


class GeneticAlgorithmToolbox:
    _toolbox: base.Toolbox
    _register: Callable[..., None]

    def __init__(self) -> None:
        self._toolbox = base.Toolbox()
        self._register = cast(
            Callable[..., None],
            getattr(self._toolbox, "register"),
        )
        self._register("clone", copy.deepcopy)

    def register_individual_factory(
        self,
        function: IndividualGenerator,
    ) -> None:
        self._register("individual", function)

    def register_population_factory(
        self,
        function: PopulationGenerator,
    ) -> None:
        self._register("population", function)

    def register_evaluate(self, function: FitnessEvaluator) -> None:
        self._register("evaluate", function)

    def register_mate(self, function: CrossoverOperator) -> None:
        self._register("mate", function)

    def register_mutate(self, function: MutationOperator) -> None:
        self._register("mutate", function)

    def register_select(self, function: SelectionOperator) -> None:
        self._register("select", function)

    def individual(self) -> Individual:
        function = cast(
            IndividualGenerator, getattr(self._toolbox, "individual")
        )
        return function()

    def population(self, n: int) -> Population:
        function = cast(
            PopulationGenerator, getattr(self._toolbox, "population")
        )
        return function(n)

    def evaluate(self, individual: Individual) -> tuple[float, ...]:
        function = cast(FitnessEvaluator, getattr(self._toolbox, "evaluate"))
        return function(individual)

    def mate(
        self,
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]:
        function = cast(CrossoverOperator, getattr(self._toolbox, "mate"))
        return function(individual1, individual2)

    def mutate(self, individual: Individual) -> tuple[Individual]:
        function = cast(MutationOperator, getattr(self._toolbox, "mutate"))
        return function(individual)

    def select(self, individuals: Population, k: int) -> Population:
        function = cast(SelectionOperator, getattr(self._toolbox, "select"))
        return function(individuals, k)

    def clone(self, obj: _T) -> _T:
        function = cast(Callable[[_T], _T], getattr(self._toolbox, "clone"))
        return function(obj)

    def map(
        self,
        function: Callable[[_T], _R],
        iterable: Iterable[_T],
    ) -> Iterable[_R]:
        map_function = cast(
            Callable[[Callable[[_T], _R], Iterable[_T]], Iterable[_R]],
            getattr(self._toolbox, "map"),
        )
        return map_function(function, iterable)


def build_toolbox(
    evaluation_function: FitnessEvaluator,
    gene_size: int,
    weights: tuple[float, ...] = (1.0,),
) -> GeneticAlgorithmToolbox:
    TypedFitness.weights = weights

    toolbox = GeneticAlgorithmToolbox()

    def individual_factory() -> Individual:
        return Individual(random.random() for _ in range(gene_size))

    toolbox.register_individual_factory(individual_factory)

    def population_factory(n: int) -> Population:
        typed_init_repeat = cast(
            Callable[
                [type[list[Individual]], Callable[[], Individual], int],
                list[Individual],
            ],
            tools.initRepeat,
        )
        return typed_init_repeat(list, toolbox.individual, n)

    toolbox.register_population_factory(population_factory)

    toolbox.register_evaluate(evaluation_function)

    toolbox.register_mate(cast(CrossoverOperator, tools.cxTwoPoint))

    def configured_mutation(ind: Individual) -> tuple[Individual]:
        typed_mut_gaussian = cast(
            Callable[[Individual, float, float, float], tuple[Individual]],
            tools.mutGaussian,
        )
        return typed_mut_gaussian(ind, 0.0, 0.1, 0.2)

    toolbox.register_mutate(configured_mutation)

    def configured_selection(individuals: Population, k: int) -> Population:
        typed_sel_tournament = cast(
            Callable[[Population, int, int], Population],
            tools.selTournament,
        )
        return typed_sel_tournament(individuals, k, 3)

    toolbox.register_select(configured_selection)

    return toolbox
