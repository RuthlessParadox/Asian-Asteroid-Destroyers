import signal
from concurrent.futures import ProcessPoolExecutor

from asian_asteroid_destroyers.genetic_algorithm import (
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
    make_checkpoint_saver,
)


def _dummy_evaluator(individual: Individual) -> tuple[float, ...]:
    """Core individual fitness evaluation function.

    Must be top-level and serializable via pickle to work with multiprocessing.
    """
    score = float(sum(x * x for x in individual))
    return (score,)


def main() -> None:
    args = get_config()
    data_directory = create_experiment_directory()

    logbook = Logbook()
    history_logger = HistoryLogger()
    statistics = Statistics()

    toolbox = build_toolbox(
        evaluation_function=_dummy_evaluator,
        gene_size=args.gene_size,
        weights=(1.0,),
    )

    emergency_saver = EmergencySaver(output_directory=data_directory)
    periodic_saver = make_checkpoint_saver(
        output_directory=data_directory,
        save_interval=args.save_interval,
    )

    _ = signal.signal(signal.SIGINT, emergency_saver.handle_interrupt)

    def combined_callback(generation: int, state: TrainingResult) -> None:
        emergency_saver.update_state(generation, state)
        periodic_saver(generation, state)

    print("Commencing evolutionary loop...")

    with ProcessPoolExecutor() as executor:
        toolbox.register("map", executor.map)

        final_result = run_simple_evolution(
            toolbox=toolbox,
            cx_pb=args.cx_pb,
            mut_pb=args.mut_pb,
            generations=args.generations,
            population_size=args.population_size,
            logbook=logbook,
            history_logger=history_logger,
            statistics=statistics,
            checkpoint_callback=combined_callback,
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
