import numpy as np
import pytest

from asian_asteroid_destroyers.evokit import (
    FitnessEvaluator,
    HallOfFame,
    HistoryLogger,
    Individual,
    Logbook,
    Population,
    Statistics,
    Toolbox,
    TrainingResult,
    build_toolbox,
    clear_fitness_cache,
    clear_individual_cache,
    get_individual_class,
    run_simple_evolution,
)


@pytest.fixture(autouse=True)
def cleanup_caches(request: pytest.FixtureRequest) -> None:
    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


@pytest.fixture
def fitness_weights() -> tuple[float, ...]:
    return (1.0,)


@pytest.fixture
def individual_class(fitness_weights: tuple[float, ...]) -> type[Individual]:
    return get_individual_class(fitness_weights)


@pytest.fixture
def dummy_evaluator() -> FitnessEvaluator:
    return lambda individual: (float(sum(individual)),)


@pytest.fixture
def configured_toolbox(
    dummy_evaluator: FitnessEvaluator,
    individual_class: type[Individual],
    fitness_weights: tuple[float, ...],
) -> Toolbox:
    gene_size = 3

    def custom_individual_factory() -> Individual:
        return individual_class([0.5, 0.5, 0.5])

    def custom_population_factory(n: int) -> list[Individual]:
        return [individual_class([0.1 * i] * gene_size) for i in range(n)]

    def mock_mate(
        individual_1: Individual,
        individual_2: Individual,
    ) -> tuple[Individual, Individual]:
        return individual_1, individual_2

    def mock_mutate(individual: Individual) -> tuple[Individual]:
        return (individual,)

    def mock_select(population: Population, k: int) -> Population:
        return list(population[:k])

    toolbox = build_toolbox(
        evaluation_function=dummy_evaluator,
        gene_size=gene_size,
        weights=fitness_weights,
        individual_factory=custom_individual_factory,
        population_factory=custom_population_factory,
        mate_operator=mock_mate,
        mutate_operator=mock_mutate,
        select_operator=mock_select,
    )
    return toolbox


def test_parameter_validation_cx_pb_bounds(
    configured_toolbox: Toolbox,
) -> None:
    with pytest.raises(ValueError, match="Crossover probability 'cx_pb'"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=-0.1,
            mut_pb=0.2,
            generations=5,
            population_size=10,
        )

    with pytest.raises(ValueError, match="Crossover probability 'cx_pb'"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=1.1,
            mut_pb=0.2,
            generations=5,
            population_size=10,
        )


def test_parameter_validation_mut_pb_bounds(
    configured_toolbox: Toolbox,
) -> None:
    with pytest.raises(ValueError, match="Mutation probability 'mut_pb'"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.5,
            mut_pb=-0.5,
            generations=5,
            population_size=10,
        )

    with pytest.raises(ValueError, match="Mutation probability 'mut_pb'"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.5,
            mut_pb=1.5,
            generations=5,
            population_size=10,
        )


def test_parameter_validation_generations(configured_toolbox: Toolbox) -> None:
    with pytest.raises(ValueError, match="Number of generations"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.2,
            mut_pb=0.2,
            generations=-1,
            population_size=10,
        )


def test_parameter_validation_population_size(
    configured_toolbox: Toolbox,
) -> None:
    with pytest.raises(ValueError, match="Population size 'population_size'"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.2,
            mut_pb=0.2,
            generations=5,
            population_size=0,
        )

    with pytest.raises(ValueError, match="Population size 'population_size'"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.2,
            mut_pb=0.2,
            generations=5,
            population_size=-5,
        )


def test_parameter_validation_start_generation_bounds(
    configured_toolbox: Toolbox,
) -> None:
    with pytest.raises(
        ValueError,
        match="Start generation 'start_generation' must be a positive integer",
    ):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.2,
            mut_pb=0.2,
            generations=5,
            population_size=10,
            start_generation=0,
        )


def test_parameter_validation_start_generation_exceeds_generations(
    configured_toolbox: Toolbox,
) -> None:
    initial_population = configured_toolbox.population(10)
    with pytest.raises(ValueError, match="cannot exceed 'generations' \\+ 1"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.2,
            mut_pb=0.2,
            generations=5,
            population_size=10,
            initial_population=initial_population,
            start_generation=7,
        )


def test_parameter_validation_midpoint_without_initial_population(
    configured_toolbox: Toolbox,
) -> None:
    with pytest.raises(
        ValueError,
        match="Initial population 'initial_population' cannot be None "
        + "with mid-point generation",
    ):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.2,
            mut_pb=0.2,
            generations=5,
            population_size=10,
            start_generation=3,
        )


def test_parameter_validation_population_size_mismatch(
    configured_toolbox: Toolbox,
) -> None:
    initial_population = configured_toolbox.population(5)
    with pytest.raises(ValueError, match="does not match 'population_size'"):
        _ = run_simple_evolution(
            configured_toolbox,
            cx_pb=0.2,
            mut_pb=0.2,
            generations=5,
            population_size=10,
            initial_population=initial_population,
            start_generation=2,
        )


def test_successful_zero_generation_run(configured_toolbox: Toolbox) -> None:
    population_size = 15
    result = run_simple_evolution(
        toolbox=configured_toolbox,
        cx_pb=0.5,
        mut_pb=0.2,
        generations=0,
        population_size=population_size,
    )
    assert isinstance(result, TrainingResult)
    assert len(result.final_population) == population_size
    assert result.logbook is None
    assert result.history_logger is None
    assert result.hall_of_fame is None


def test_successful_multi_generation_run(configured_toolbox: Toolbox) -> None:
    generations_count = 2
    population_size = 20
    result = run_simple_evolution(
        toolbox=configured_toolbox,
        cx_pb=0.2,
        mut_pb=0.1,
        generations=generations_count,
        population_size=population_size,
        seed=42,
    )
    assert len(result.final_population) == population_size
    for individual in result.final_population:
        assert isinstance(individual, Individual)
        assert individual.fitness.valid
        assert isinstance(individual.fitness.values, tuple)


def test_seed_as_generator_instance(configured_toolbox: Toolbox) -> None:
    rng = np.random.default_rng(12345)
    population_size = 8
    result = run_simple_evolution(
        toolbox=configured_toolbox,
        cx_pb=0.9,
        mut_pb=0.9,
        generations=1,
        population_size=population_size,
        seed=rng,
    )
    assert len(result.final_population) == population_size


def test_optional_components_recording(configured_toolbox: Toolbox) -> None:
    hall_of_fame = HallOfFame(max_size=2)
    stats = Statistics()
    logbook = Logbook()
    history_logger = HistoryLogger()
    population_size = 12

    result = run_simple_evolution(
        toolbox=configured_toolbox,
        cx_pb=0.5,
        mut_pb=0.5,
        generations=2,
        population_size=population_size,
        hall_of_fame=hall_of_fame,
        statistics=stats,
        logbook=logbook,
        history_logger=history_logger,
    )

    assert result.hall_of_fame is hall_of_fame
    assert result.logbook is logbook
    assert result.history_logger is history_logger

    assert len(hall_of_fame) > 0
    assert not logbook.dataframe.is_empty()
    assert not history_logger.dataframe.is_empty()


def test_checkpoint_callback_execution(configured_toolbox: Toolbox) -> None:
    captured_generations: list[int] = []
    population_size = 10

    def sample_callback(generation: int, current_res: TrainingResult) -> None:
        captured_generations.append(generation)
        assert isinstance(current_res, TrainingResult)
        assert len(current_res.final_population) == population_size

    _ = run_simple_evolution(
        toolbox=configured_toolbox,
        cx_pb=0.1,
        mut_pb=0.1,
        generations=3,
        population_size=population_size,
        checkpoint_callback=sample_callback,
    )

    assert captured_generations == [0, 1, 2, 3]


def test_resumption_skips_generation_zero_and_runs_correct_generations(
    configured_toolbox: Toolbox,
) -> None:
    population_size = 10
    initial_population = configured_toolbox.population(population_size)
    captured_generations: list[int] = []

    def sample_callback(
        generation: int,
        _current_state: TrainingResult,
    ) -> None:
        captured_generations.append(generation)

    _ = run_simple_evolution(
        toolbox=configured_toolbox,
        cx_pb=0.1,
        mut_pb=0.1,
        generations=4,
        population_size=population_size,
        checkpoint_callback=sample_callback,
        initial_population=initial_population,
        start_generation=3,
    )

    assert captured_generations == [3, 4]
