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
        fitness_type = type(
            "DynamicReweightedFitness",
            (individual.fitness.__class__,),
            {"weights": new_weights},
        )

        has_valid_values = hasattr(individual.fitness, "values") and getattr(
            individual.fitness, "valid", False
        )

        raw_values: tuple[float, ...] | None = (
            cast(tuple[float, ...], getattr(individual.fitness, "values"))
            if has_valid_values
            else None
        )

        individual.fitness.__class__ = fitness_type

        for attr in ("values", "wvalues"):
            if hasattr(individual.fitness, attr):
                delattr(individual.fitness, attr)

        if raw_values is not None:
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


class GeneticAlgorithmToolbox(base.Toolbox, DeapToolboxProtocol):
    @override
    def register(
        self,
        alias: str,
        method: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> None:
        base_register = cast(
            Callable[..., None],
            getattr(base.Toolbox, "register"),
        )
        base_register(self, alias, method, *args, **kwargs)

    @override
    def unregister(self, alias: str) -> None:
        base_unregister = cast(
            Callable[[object, str], None],
            getattr(base.Toolbox, "unregister"),
        )
        base_unregister(self, alias)

    @override
    def individual(self) -> Individual:
        return cast(Individual, getattr(self, "individual")())

    @override
    def population(self, n: int) -> Population:
        return cast(Population, getattr(self, "population")(n))

    @override
    def evaluate(self, individual: Individual) -> tuple[float, ...]:
        return cast(tuple[float, ...], getattr(self, "evaluate")(individual))

    @override
    def mate(
        self,
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]:
        return cast(
            tuple[Individual, Individual],
            getattr(self, "mate")(individual1, individual2),
        )

    @override
    def mutate(self, individual: Individual) -> tuple[Individual]:
        return cast(tuple[Individual], getattr(self, "mutate")(individual))

    @override
    def select(self, individuals: Population, k: int) -> Population:
        return cast(Population, getattr(self, "select")(individuals, k))

    @override
    def clone(self, obj: _T) -> _T:
        return cast(_T, getattr(self, "clone")(obj))

    @override
    def map(
        self,
        function: Callable[[_T], _R],
        iterable: Iterable[_T],
    ) -> Iterable[_R]:
        return cast(Iterable[_R], getattr(self, "map")(function, iterable))


def build_toolbox(
    evaluation_function: FitnessEvaluator,
    gene_size: int,
    weights: tuple[float, ...] = (1.0,),
) -> GeneticAlgorithmToolbox:
    individual_class = create_individual_class(weights)
    toolbox = GeneticAlgorithmToolbox()
    toolbox.register("clone", copy.deepcopy)

    def individual_factory() -> Individual:
        return individual_class(random.random() for _ in range(gene_size))

    toolbox.register("individual", individual_factory)

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

    toolbox.register("population", population_factory)
    toolbox.register("evaluate", evaluation_function)
    toolbox.register("mate", cast(CrossoverOperator, tools.cxTwoPoint))
    toolbox.register("mutate", configured_mutation)
    toolbox.register("select", configured_selection)

    return toolbox
