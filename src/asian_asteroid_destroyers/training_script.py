import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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
    make_checkpoint_saver,
)


def dummy_evaluator(individual: Individual) -> tuple[float, ...]:
    """Core individual fitness evaluation function.

    Must be top-level and serializable via pickle to work with multiprocessing.
    """
    score = float(sum(x * x for x in individual))
    return (score,)


def main() -> None:
    generations = 100
    population_size = 100
    checkpoint_directory = Path("./checkpoints")
    save_every_n_generations = 10

    logbook = Logbook()
    history_logger = HistoryLogger()
    statistics = Statistics()

    toolbox = build_toolbox(
        evaluation_function=dummy_evaluator,
        gene_size=50,
        weights=(1.0,),
    )

    emergency_saver = EmergencySaver(output_directory=checkpoint_directory)
    periodic_saver = make_checkpoint_saver(
        output_directory=checkpoint_directory,
        save_interval=save_every_n_generations,
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
            cx_pb=0.6,
            mut_pb=0.2,
            generations=generations,
            population_size=population_size,
            logbook=logbook,
            history_logger=history_logger,
            statistics=statistics,
            checkpoint_callback=combined_callback,
        )

    print("\nEvolutionary horizon run successfully finished.")

    final_logbook = final_result.logbook
    if final_logbook is not None:
        final_logbook.save_parquet(
            checkpoint_directory / "final_logbook.parquet",
        )
    final_history_logger = final_result.history_logger
    if final_history_logger is not None:
        final_history_logger.save_parquet(
            checkpoint_directory / "final_history.parquet",
        )
    print("Final run assets secured to disk.")


if __name__ == "__main__":
    main()
