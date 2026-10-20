import copy
import random
import sys
from collections.abc import Callable, Iterable
from typing import NoReturn, TypeVar, cast

from deap import base, tools
from typing_extensions import override

from asian_asteroid_destroyers.genetic_algorithm.operators import (
    crossover_two_point,
    mutation_gaussian,
    selection_tournament,
)
from asian_asteroid_destroyers.genetic_algorithm.types import (
    CloneMethod,
    CrossoverOperator,
    FitnessEvaluator,
    Individual,
    IndividualGenerator,
    MapMethod,
    MutationOperator,
    Population,
    PopulationGenerator,
    SelectionOperator,
    TypedFitness,
)

_T = TypeVar("_T")
_R = TypeVar("_R")


_FITNESS_CLASS_CACHE: dict[tuple[float, ...], type[TypedFitness]] = {}
_INDIVIDUAL_CLASS_CACHE: dict[tuple[float, ...], type[Individual]] = {}


def _get_or_create_fitness_class(
    weights: tuple[float, ...],
) -> type[TypedFitness]:
    if weights in _FITNESS_CLASS_CACHE:
        return _FITNESS_CLASS_CACHE[weights]
    clean_weight_string = "_".join(
        str(weight).replace(".", "_").replace("-", "neg") for weight in weights
    )
    class_name = f"DynamicFitness_{clean_weight_string}"
    new_class = cast(
        type[TypedFitness],
        type(class_name, (TypedFitness,), {"weights": weights}),
    )
    current_module = sys.modules[__name__]
    setattr(current_module, class_name, new_class)
    new_class.__module__ = __name__
    _FITNESS_CLASS_CACHE[weights] = new_class
    return new_class


def _create_individual_class(weights: tuple[float, ...]) -> type[Individual]:
    if weights in _INDIVIDUAL_CLASS_CACHE:
        return _INDIVIDUAL_CLASS_CACHE[weights]
    fitness_type = _get_or_create_fitness_class(weights)
    clean_weight_string = "_".join(
        str(weight).replace(".", "_").replace("-", "neg") for weight in weights
    )
    class_name = f"DynamicIndividual_{clean_weight_string}"

    def individual_init(
        self: Individual,
        iterable: Iterable[float] | None = None,
    ) -> None:
        super(Individual, self).__init__(iterable or [])
        self.fitness = fitness_type()

    new_class = cast(
        type[Individual],
        type(class_name, (Individual,), {"__init__": individual_init}),
    )
    current_module = sys.modules[__name__]
    setattr(current_module, class_name, new_class)
    new_class.__module__ = __name__
    _INDIVIDUAL_CLASS_CACHE[weights] = new_class
    return new_class


def reweight_population(
    population: Population,
    new_weights: tuple[float, ...],
) -> None:
    new_fitness_type = _get_or_create_fitness_class(new_weights)
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
    individual_class = _create_individual_class(weights)
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
