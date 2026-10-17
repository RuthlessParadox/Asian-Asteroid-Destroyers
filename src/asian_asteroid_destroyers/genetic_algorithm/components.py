import copy
import random
from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar, cast, override

from deap import base, tools

_T = TypeVar("_T")
_R = TypeVar("_R")


class TypedFitness(base.Fitness):
    weights: tuple[float, ...] = ()
    values: tuple[float, ...]

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


def create_individual_class(weights: tuple[float, ...]) -> type[Individual]:
    fitness_type = type(
        "RuntimeFitness",
        (TypedFitness,),
        {"weights": weights},
    )

    class ContextIndividual(Individual):
        fitness: TypedFitness

        def __init__(self, iterable: Iterable[float] | None = None) -> None:
            super().__init__(iterable)
            self.fitness = fitness_type()

    return ContextIndividual


Population = list[Individual]
IndividualGenerator = Callable[[], Individual]
PopulationGenerator = Callable[[int], Population]
FitnessEvaluator = Callable[[Individual], tuple[float, ...]]
CrossoverOperator = Callable[
    [Individual, Individual], tuple[Individual, Individual]
]
MutationOperator = Callable[[Individual], tuple[Individual]]
SelectionOperator = Callable[[Population, int], Population]


def reweight_population(
    population: Population,
    new_weights: tuple[float, ...],
) -> None:
    for individual in population:
        individual.fitness.__class__.weights = new_weights

        if hasattr(individual.fitness, "values") and individual.fitness.valid:
            raw_values = individual.fitness.values
            del individual.fitness.values
            del individual.fitness.wvalues
            individual.fitness.values = raw_values


class DeapToolboxProtocol(Protocol):
    def register(
        self,
        alias: str,
        method: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> None: ...
    def unregister(self, alias: str) -> None: ...
    def individual(self) -> Individual: ...
    def population(self, n: int) -> Population: ...
    def evaluate(self, individual: Individual) -> tuple[float, ...]: ...
    def mate(
        self,
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]: ...
    def mutate(self, individual: Individual) -> tuple[Individual]: ...
    def select(self, individuals: Population, k: int) -> Population: ...
    def clone(self, obj: _T) -> _T: ...
    def map(
        self,
        function: Callable[[_T], _R],
        iterable: Iterable[_T],
    ) -> Iterable[_R]: ...


class TypedDeapToolbox(base.Toolbox, DeapToolboxProtocol):
    @override
    def individual(self) -> Individual:
        raise NotImplementedError

    @override
    def population(self, n: int) -> Population:
        raise NotImplementedError

    @override
    def evaluate(self, individual: Individual) -> tuple[float, ...]:
        raise NotImplementedError

    @override
    def mate(
        self,
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]:
        raise NotImplementedError

    @override
    def mutate(self, individual: Individual) -> tuple[Individual]:
        raise NotImplementedError

    @override
    def select(self, individuals: Population, k: int) -> Population:
        raise NotImplementedError

    @override
    def clone(self, obj: object) -> object:
        raise NotImplementedError

    @override
    def map(
        self,
        function: Callable[..., object],
        iterable: Iterable[object],
    ) -> Iterable[object]:
        raise NotImplementedError


class GeneticAlgorithmToolbox:
    deap_toolbox: DeapToolboxProtocol

    def __init__(self) -> None:
        self.deap_toolbox = cast(DeapToolboxProtocol, TypedDeapToolbox())
        self.deap_toolbox.register("clone", copy.deepcopy)

    def register_individual_factory(
        self,
        function: IndividualGenerator,
    ) -> None:
        self.deap_toolbox.register("individual", function)

    def register_population_factory(
        self,
        function: PopulationGenerator,
    ) -> None:
        self.deap_toolbox.register("population", function)

    def register_evaluate(self, function: FitnessEvaluator) -> None:
        self.deap_toolbox.register("evaluate", function)

    def register_mate(self, function: CrossoverOperator) -> None:
        self.deap_toolbox.register("mate", function)

    def register_mutate(self, function: MutationOperator) -> None:
        self.deap_toolbox.register("mutate", function)

    def register_select(self, function: SelectionOperator) -> None:
        self.deap_toolbox.register("select", function)

    def individual(self) -> Individual:
        return self.deap_toolbox.individual()

    def population(self, n: int) -> Population:
        return self.deap_toolbox.population(n)

    def evaluate(self, individual: Individual) -> tuple[float, ...]:
        return self.deap_toolbox.evaluate(individual)

    def mate(
        self,
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]:
        return self.deap_toolbox.mate(individual1, individual2)

    def mutate(self, individual: Individual) -> tuple[Individual]:
        return self.deap_toolbox.mutate(individual)

    def select(self, individuals: Population, k: int) -> Population:
        return self.deap_toolbox.select(individuals, k)

    def clone(self, obj: _T) -> _T:
        return self.deap_toolbox.clone(obj)

    def map(
        self,
        function: Callable[[_T], _R],
        iterable: Iterable[_T],
    ) -> Iterable[_R]:
        return self.deap_toolbox.map(function, iterable)


def build_toolbox(
    evaluation_function: FitnessEvaluator,
    gene_size: int,
    weights: tuple[float, ...] = (1.0,),
) -> GeneticAlgorithmToolbox:
    individual_class = create_individual_class(weights)
    toolbox = GeneticAlgorithmToolbox()

    def individual_factory() -> Individual:
        return individual_class(random.random() for _ in range(gene_size))

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

    def configured_mutation(ind: Individual) -> tuple[Individual]:
        typed_mut_gaussian = cast(
            Callable[[Individual, float, float, float], tuple[Individual]],
            tools.mutGaussian,
        )
        return typed_mut_gaussian(ind, 0.0, 0.1, 0.2)

    def configured_selection(individuals: Population, k: int) -> Population:
        typed_sel_tournament = cast(
            Callable[[Population, int, int], Population],
            tools.selTournament,
        )
        return typed_sel_tournament(individuals, k, 3)

    toolbox.register_population_factory(population_factory)
    toolbox.register_evaluate(evaluation_function)
    toolbox.register_mate(cast(CrossoverOperator, tools.cxTwoPoint))
    toolbox.register_mutate(configured_mutation)
    toolbox.register_select(configured_selection)

    return toolbox
