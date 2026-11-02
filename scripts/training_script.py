import pathlib
import signal

import dill
import numpy as np
from pathos.multiprocessing import ProcessPool

from asian_asteroid_destroyers.evokit import (
    HistoryLogger,
    Individual,
    Logbook,
    Statistics,
    TrainingResult,
    build_toolbox,
    run_simple_evolution,
)
from asian_asteroid_destroyers.infrastructure import (
    EmergencySaver,
    create_experiment_directory,
    get_config,
    load_resumption_workspace,
    make_checkpoint_saver,
)


def _dummy_evaluator(individual: Individual) -> tuple[float, ...]:
    """Core individual fitness evaluation function.

    Must be top-level and serializable via pickle to work with multiprocessing.
    """
    score = float(sum(x * x for x in individual))
    return (score,)


def main() -> None:
    dill.settings["recurse"] = True
    args = get_config()

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
    else:
        print("Commencing a fresh evolutionary workspace...")
        data_directory = create_experiment_directory()
        algorithm = run_simple_evolution
        start_generation = 1
        rng = np.random.default_rng()
        logbook = Logbook()
        history_logger = HistoryLogger()
        population = None

    statistics = Statistics()
    toolbox = build_toolbox(
        evaluation_function=_dummy_evaluator,
        gene_size=args.gene_size,
        weights=(1.0,),
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

    print("Commencing evolutionary loop...")
    with ProcessPool() as executor:
        toolbox.register("map", executor.map)

        final_result = algorithm(
            toolbox=toolbox,
            cx_pb=args.cx_pb,
            mut_pb=args.mut_pb,
            generations=args.generations,
            population_size=args.population_size,
            seed=rng,
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
