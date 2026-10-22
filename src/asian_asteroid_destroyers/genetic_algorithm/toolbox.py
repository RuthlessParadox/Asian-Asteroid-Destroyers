import copy
import random
from collections.abc import Callable
from typing import NoReturn, cast

from deap import base, tools
from typing_extensions import override

from .crossover import cx_two_point
from .individual import Individual, get_individual_class
from .mutation import mut_gaussian
from .selection import sel_tournament
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


def _raise_unregistered(name: str) -> NoReturn:
    raise NotImplementedError(
        f"The DEAP operation '{name}' has not been registered yet."
    )


class Toolbox(base.Toolbox):
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
    toolbox: Toolbox,
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
    return cx_two_point()


def _default_mut() -> MutationOperator:
    return mut_gaussian(0.0, 0.1, 0.2)


def _default_sel() -> SelectionOperator:
    return sel_tournament(3)


def build_toolbox(
    evaluation_function: FitnessEvaluator,
    gene_size: int,
    weights: tuple[float, ...] = (1.0,),
    individual_factory: IndividualGenerator | None = None,
    population_factory: PopulationGenerator | None = None,
    mate_operator: CrossoverOperator | None = None,
    mutate_operator: MutationOperator | None = None,
    select_operator: SelectionOperator | None = None,
) -> Toolbox:
    individual_class = get_individual_class(weights)
    toolbox = Toolbox()

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
    toolbox.register("mutate", mutate_operator or _default_mut())
    toolbox.register("select", select_operator or _default_sel())

    return toolbox
