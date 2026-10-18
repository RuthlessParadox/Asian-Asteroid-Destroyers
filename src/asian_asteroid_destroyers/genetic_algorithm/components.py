import copy
import random
from collections.abc import Callable, Iterable
from typing import NoReturn, Protocol, TypeVar, cast, override

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
        new_fitness_type = type(
            "DynamicReweightedFitness",
            (TypedFitness,),
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

        individual.fitness = new_fitness_type()
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

    individual: IndividualGenerator
    population: PopulationGenerator
    evaluate: FitnessEvaluator
    mate: CrossoverOperator
    mutate: MutationOperator
    select: SelectionOperator

    def clone(self, obj: _T) -> _T: ...
    def map(
        self,
        function: Callable[[_T], _R],
        iterable: Iterable[_T],
    ) -> Iterable[_R]: ...


def _unregistered_stub() -> NoReturn:
    raise NotImplementedError(
        "Method has not been registered on the toolbox yet."
    )


class GeneticAlgorithmToolbox(base.Toolbox, DeapToolboxProtocol):
    def __init__(self) -> None:
        super().__init__()

        self.individual: IndividualGenerator = lambda: _unregistered_stub()
        self.population: PopulationGenerator = lambda n: _unregistered_stub()
        self.evaluate: FitnessEvaluator = lambda ind: _unregistered_stub()
        self.mate: CrossoverOperator = lambda ind1, ind2: _unregistered_stub()
        self.mutate: MutationOperator = lambda ind: _unregistered_stub()
        self.select: SelectionOperator = lambda individuals, k: (
            _unregistered_stub()
        )

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
    def clone(self, obj: _T) -> _T:
        clone_func = cast(Callable[[_T], _T], self.__dict__["clone"])
        return clone_func(obj)

    @override
    def map(
        self,
        function: Callable[[_T], _R],
        iterable: Iterable[_T],
    ) -> Iterable[_R]:
        map_func = cast(
            Callable[[Callable[[_T], _R], Iterable[_T]], Iterable[_R]],
            self.__dict__["map"],
        )
        return map_func(function, iterable)


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

    def configured_mate(
        individual1: Individual,
        individual2: Individual,
    ) -> tuple[Individual, Individual]:
        typed_cx = cast(
            Callable[[Individual, Individual], tuple[Individual, Individual]],
            tools.cxTwoPoint,
        )
        return typed_cx(individual1, individual2)

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
    toolbox.register("mate", configured_mate)
    toolbox.register("mutate", configured_mutation)
    toolbox.register("select", configured_selection)

    return toolbox
