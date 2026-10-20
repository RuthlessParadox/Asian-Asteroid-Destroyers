import copy
import random
from collections.abc import Callable
from typing import NoReturn, TypeVar, cast

from deap import base, tools
from typing_extensions import override

from .fitness_class_cache import get_fitness_class
from .individual_class_cache import get_individual_class
from .individuals import Individual
from .operators import (
    crossover_two_point,
    mutation_gaussian,
    selection_tournament,
)
from .types import (
    CloneMethod,
    CrossoverOperator,
    FitnessEvaluator,
    IndividualGenerator,
    MapMethod,
    MutationOperator,
    Population,
    PopulationGenerator,
    SelectionOperator,
)

_T = TypeVar("_T")
_R = TypeVar("_R")


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


def _raise_unregistered(name: str) -> NoReturn:
    raise NotImplementedError(
        f"The DEAP operation '{name}' has not been registered yet."
    )


class GeneticAlgorithmToolbox(base.Toolbox):
    clone: CloneMethod
    map: MapMethod
    individual: IndividualGenerator
    population: PopulationGenerator
    evaluate: FitnessEvaluator
    mate: CrossoverOperator
    mutate: MutationOperator
    select: SelectionOperator

    def __init__(self) -> None:
        super().__init__()

        base_register = cast(
            Callable[..., None], getattr(base.Toolbox, "register")
        )
        base_register(self, "clone", copy.deepcopy)
        self.clone = cast(CloneMethod, getattr(self, "clone"))
        base_register(self, "map", map)
        self.map = cast(MapMethod, getattr(self, "map"))

        def _stub_factory(name: str) -> Callable[..., NoReturn]:
            return lambda *args, **kwargs: _raise_unregistered(name)

        factory: Callable[[str], Callable[..., NoReturn]] = _stub_factory
        self.individual = cast(IndividualGenerator, factory("individual"))
        self.population = cast(PopulationGenerator, factory("population"))
        self.evaluate = cast(FitnessEvaluator, factory("evaluate"))
        self.mate = cast(CrossoverOperator, factory("mate"))
        self.mutate = cast(MutationOperator, factory("mutate"))
        self.select = cast(SelectionOperator, factory("select"))

    @override
    def register(
        self,
        alias: str,
        function: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> None:
        base_register = cast(
            Callable[..., None],
            getattr(base.Toolbox, "register"),
        )
        base_register(self, alias, function, *args, **kwargs)
        if hasattr(self, alias):
            setattr(
                self, alias, cast(Callable[..., object], getattr(self, alias))
            )

    @override
    def unregister(self, alias: str) -> None:
        base_unregister = cast(
            Callable[[object, str], None],
            getattr(base.Toolbox, "unregister"),
        )
        base_unregister(self, alias)
        if hasattr(self, alias):
            delattr(self, alias)


def _get_default_individual_factory(
    individual_class: type[Individual],
    gene_size: int,
) -> IndividualGenerator:
    def default_individual_factory() -> Individual:
        return individual_class(random.random() for _ in range(gene_size))

    return default_individual_factory


def _get_default_population_factory(
    toolbox: GeneticAlgorithmToolbox,
) -> PopulationGenerator:
    def default_population_factory(n: int) -> Population:
        typed_init_repeat = cast(
            Callable[
                [type[list[Individual]], Callable[[], Individual], int],
                list[Individual],
            ],
            tools.initRepeat,
        )
        return typed_init_repeat(list, toolbox.individual, n)

    return default_population_factory


def _default_mate() -> CrossoverOperator:
    return crossover_two_point()


def _default_mutation() -> MutationOperator:
    return mutation_gaussian(0.0, 0.1, 0.2)


def _default_selection() -> SelectionOperator:
    return selection_tournament(3)


def build_toolbox(
    evaluation_function: FitnessEvaluator,
    gene_size: int,
    weights: tuple[float, ...] = (1.0,),
    individual_factory: IndividualGenerator | None = None,
    population_factory: PopulationGenerator | None = None,
    mate_operator: CrossoverOperator | None = None,
    mutate_operator: MutationOperator | None = None,
    select_operator: SelectionOperator | None = None,
) -> GeneticAlgorithmToolbox:
    individual_class = get_individual_class(weights)
    toolbox = GeneticAlgorithmToolbox()

    toolbox.register("evaluate", evaluation_function)
    toolbox.register(
        "individual",
        individual_factory
        or _get_default_individual_factory(individual_class, gene_size),
    )
    toolbox.register(
        "population",
        population_factory or _get_default_population_factory(toolbox),
    )
    toolbox.register("mate", mate_operator or _default_mate())
    toolbox.register("mutate", mutate_operator or _default_mutation())
    toolbox.register("select", select_operator or _default_selection())

    return toolbox
