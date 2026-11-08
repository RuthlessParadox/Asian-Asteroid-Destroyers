from __future__ import annotations

import pathlib
import signal
import sys
from typing import TYPE_CHECKING

import dill
import numpy as np

from asian_asteroid_destroyers.chromosomes import (
    create_geometry_repair_maps,
    flatten_chromosome,
    get_default_controller_template,
    repair_geometry_invariants,
)
from asian_asteroid_destroyers.evaluation import FuzzyFitnessEvaluator
from asian_asteroid_destroyers.evokit import (
    HallOfFame,
    HistoryLogger,
    Logbook,
    Statistics,
    build_toolbox,
    cx_fuzzy_blend,
    get_individual_class,
    mut_fuzzy_gaussian,
    run_nsga_evolution,
    sel_nsga_3,
)
from asian_asteroid_destroyers.infrastructure import (
    EmergencySaver,
    create_experiment_directory,
    get_trainer_config,
    load_resumption_workspace,
    make_checkpoint_saver,
)

if TYPE_CHECKING:
    from asian_asteroid_destroyers.evokit import Individual, TrainingResult

GLOBAL_EVALUATOR_INSTANCE: FuzzyFitnessEvaluator | None = None


class _MultiprocessingEvaluationWorker:
    __slots__: tuple[str, ...] = ("evaluator",)
    evaluator: FuzzyFitnessEvaluator

    def __init__(self, evaluator: FuzzyFitnessEvaluator) -> None:
        self.evaluator = evaluator

    def __call__(self, individual: Individual) -> tuple[float, ...]:
        return self.evaluator.evaluate_individual(individual)


def main() -> None:
    dill.settings["recurse"] = True
    args = get_trainer_config()

    template = get_default_controller_template()
    flat_genes_length = len(flatten_chromosome(template))
    weights = (1.0, 1.0, 1.0, -1.0, -1.0)

    if args.resume is not None:
        data_directory = pathlib.Path(args.resume)
        print(f"Resuming evolution context from: {data_directory.name}")

        workspace = load_resumption_workspace(data_directory)

        if args.generations <= workspace.last_generation:
            raise ValueError(
                "Horizon Conflict: Your target limit "
                + f"'--generations {args.generations}' has already been "
                + "reached or exceeded by this checkpoint (Current progress: "
                + f"generation {workspace.last_generation}). Please supply a "
                + "higher '--generations' value to extend training."
            )

        algorithm = workspace.algorithm
        start_generation = workspace.last_generation + 1
        rng = workspace.rng

        logbook = workspace.saved_state.logbook
        history_logger = workspace.saved_state.history_logger
        population = workspace.saved_state.final_population
        hall_of_fame = workspace.saved_state.hall_of_fame
    else:
        print("Commencing a fresh evolutionary workspace...")
        data_directory = create_experiment_directory()
        algorithm = run_nsga_evolution
        start_generation = 1
        rng = np.random.default_rng()
        logbook = Logbook()
        history_logger = HistoryLogger()
        hall_of_fame = HallOfFame()
        population = None

    initial_seed = int(rng.integers(1, 1000000))
    runtime_evaluator = FuzzyFitnessEvaluator(
        template=template,
        profile_path=args.scenarios_path,
        evaluation_seed=initial_seed,
    )
    setattr(
        sys.modules["__main__"],
        "GLOBAL_EVALUATOR_INSTANCE",
        runtime_evaluator,
    )

    triangular_array, trapezoidal_array = create_geometry_repair_maps(template)
    individual_class = get_individual_class(weights)

    def fuzzy_individual_factory() -> Individual:
        raw_random_genes = [
            float(rng.uniform(-1.0, 1.0)) for _ in range(flat_genes_length)
        ]
        valid_genes = repair_geometry_invariants(
            raw_random_genes,
            triangular_array,
            trapezoidal_array,
        )
        return individual_class(valid_genes)

    mate_operator = cx_fuzzy_blend(template=template, rng=rng, alpha=0.25)
    mutate_operator = mut_fuzzy_gaussian(
        template=template,
        rng=rng,
        mu=0.0,
        sigma=0.04,
        independent_probability=0.08,
    )

    select_operator = sel_nsga_3(num_objectives=5, subdivisions=4)

    statistics = Statistics()

    evaluation_worker = _MultiprocessingEvaluationWorker(runtime_evaluator)

    toolbox = build_toolbox(
        evaluation_function=evaluation_worker,
        gene_size=flat_genes_length,
        weights=weights,
        individual_factory=fuzzy_individual_factory,
        mate_operator=mate_operator,
        mutate_operator=mutate_operator,
        select_operator=select_operator,
    )

    emergency_saver = EmergencySaver(
        output_directory=data_directory,
        rng=rng,
        algorithm=algorithm,
    )
    periodic_saver = make_checkpoint_saver(
        output_directory=data_directory,
        rng=rng,
        algorithm=algorithm,
        save_interval=args.save_interval,
    )

    _ = signal.signal(signal.SIGINT, emergency_saver.handle_interrupt)

    def combined_callback(generation: int, state: TrainingResult) -> None:
        emergency_saver.update_state(generation, state)
        periodic_saver(generation, state)

        next_generation_seed = int(rng.integers(1, 1000000))
        runtime_evaluator.evaluation_seed = next_generation_seed

    print("Commencing evolutionary loop...")
    final_result = algorithm(
        toolbox=toolbox,
        cx_pb=args.cx_pb,
        mut_pb=args.mut_pb,
        generations=args.generations,
        population_size=args.population_size,
        seed=rng,
        hall_of_fame=hall_of_fame,
        logbook=logbook,
        history_logger=history_logger,
        statistics=statistics,
        checkpoint_callback=combined_callback,
        initial_population=population,
        start_generation=start_generation,
    )

    print("\nEvolutionary horizon run successfully finished.")

    final_logbook = final_result.logbook
    if final_logbook is not None:
        final_logbook.save_parquet(
            data_directory / "final_logbook.parquet",
        )
    final_history_logger = final_result.history_logger
    if final_history_logger is not None:
        final_history_logger.save_parquet(
            data_directory / "final_history.parquet",
        )
    print("Final run assets secured to disk.")


if __name__ == "__main__":
    main()
