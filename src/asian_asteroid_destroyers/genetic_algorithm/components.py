import copy
import random
import sys
from collections.abc import Callable, Iterable
from typing import NoReturn, TypeVar, cast, override

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
        if not hasattr(self, "fitness"):
            self.fitness = TypedFitness()


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


def _unregistered_stub(*_args: object, **_kwargs: object) -> NoReturn:
    raise NotImplementedError("This DEAP method has not been registered yet.")


class GeneticAlgorithmToolbox(base.Toolbox):
    individual: IndividualGenerator
    population: PopulationGenerator
    evaluate: FitnessEvaluator
    mate: CrossoverOperator
    mutate: MutationOperator
    select: SelectionOperator

    def __init__(self) -> None:
        super().__init__()
        self.individual = _unregistered_stub
        self.population = _unregistered_stub
        self.evaluate = _unregistered_stub
        self.mate = _unregistered_stub
        self.mutate = _unregistered_stub
        self.select = _unregistered_stub

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


def build_toolbox(
    evaluation_function: FitnessEvaluator,
    gene_size: int,
    weights: tuple[float, ...] = (1.0,),
) -> GeneticAlgorithmToolbox:
    individual_class = _create_individual_class(weights)

    toolbox = GeneticAlgorithmToolbox()

    toolbox.register("clone", copy.deepcopy)
    toolbox.register("map", map)

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

    def configured_mutation(individual: Individual) -> tuple[Individual]:
        typed_mut_gaussian = cast(
            Callable[[Individual, float, float, float], tuple[Individual]],
            tools.mutGaussian,
        )
        return typed_mut_gaussian(individual, 0.0, 0.1, 0.2)

    def configured_selection(population: Population, k: int) -> Population:
        typed_sel_tournament = cast(
            Callable[[Population, int, int], Population], tools.selTournament
        )
        return typed_sel_tournament(population, k, 3)

    toolbox.register("population", population_factory)
    toolbox.register("evaluate", evaluation_function)
    toolbox.register("mate", configured_mate)
    toolbox.register("mutate", configured_mutation)
    toolbox.register("select", configured_selection)

    return toolbox
