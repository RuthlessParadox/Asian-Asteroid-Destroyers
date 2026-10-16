import copy
import random
from collections.abc import Callable, Iterable
from typing import TypeVar, cast

from deap import base, tools

_T = TypeVar("_T")
_R = TypeVar("_R")


class TypedFitness(base.Fitness):
    weights: tuple[float, ...] = (1.0,)

    def __init__(self, values: tuple[float, ...] | None = None) -> None:
        super().__init__()
        if values is not None:
            self.values: tuple[float, ...] = values


class Individual(list[float]):
    def __init__(self, iterable: Iterable[float] | None = None) -> None:
        if iterable is not None:
            super().__init__(iterable)
        else:
            super().__init__()
        self.fitness: TypedFitness = TypedFitness()


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
    def __init__(self) -> None:
        self._toolbox: base.Toolbox = base.Toolbox()
        register_function = cast(
            Callable[..., None], getattr(self._toolbox, "register")
        )
        register_function("clone", copy.deepcopy)

    def register_operator(
        self,
        name: str,
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> None:
        register_function = cast(
            Callable[..., None], getattr(self._toolbox, "register")
        )
        register_function(name, function, *args, **kwargs)

    def get_operator(self, name: str) -> Callable[..., object]:
        return cast(Callable[..., object], getattr(self._toolbox, name))

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
        self, ind1: Individual, ind2: Individual
    ) -> tuple[Individual, Individual]:
        function = cast(CrossoverOperator, getattr(self._toolbox, "mate"))
        return function(ind1, ind2)

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
        self, function: Callable[[_T], _R], iterable: Iterable[_T]
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
    wrapper = GeneticAlgorithmToolbox()

    def individual_factory() -> Individual:
        genes = [random.random() for _ in range(gene_size)]
        return Individual(genes)

    wrapper.register_operator("individual", individual_factory)

    initialize_repeat_population = cast(
        Callable[
            [type[list[Individual]], Callable[[], Individual], int], Population
        ],
        tools.initRepeat,
    )

    wrapper.register_operator(
        "population",
        initialize_repeat_population,
        list,
        cast(IndividualGenerator, wrapper.get_operator("individual")),
    )

    wrapper.register_operator("evaluate", evaluation_function)

    wrapper.register_operator(
        "mate", cast(CrossoverOperator, tools.cxTwoPoint)
    )

    mutate_function = cast(MutationOperator, cast(object, tools.mutGaussian))
    wrapper.register_operator(
        "mutate",
        mutate_function,
        mu=0.0,
        sigma=0.1,
        indpb=0.2,
    )

    select_function = cast(
        SelectionOperator, cast(object, tools.selTournament)
    )
    wrapper.register_operator(
        "select",
        select_function,
        tournsize=3,
    )

    return wrapper
