from __future__ import annotations

import pathlib
import signal
from typing import TYPE_CHECKING

import dill
import numpy as np
from pathos.multiprocessing import ProcessPool

from asian_asteroid_destroyers.chromosomes import (
    ControllerChromosome,
    MFParamTrapezoidal,
    MFParamTriangular,
    RuleGene,
    SubsystemChromosome,
    flatten_chromosome,
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
    run_simple_evolution,
    sel_tournament,
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


def _make_structural_template() -> ControllerChromosome:
    """Creates a comprehensive 3-input structural template for the XFC.

    Each input axis contains three linguistic terms (Low, Medium, High).
    The rules map every dimension to form a legible, cohesive initial strategy.
    """
    # -------------------------------------------------------------------------
    # Pool Definition: 3 Shapes per Subsystem Input
    # Pool layout index dictionary:
    #   Index [0, 1, 2] -> Input 1 Terms (Low, Med, High)
    #   Index [3, 4, 5] -> Input 2 Terms (Low, Med, High)
    #   Index [6, 7, 8] -> Input 3 Terms (Low, Med, High)
    # -------------------------------------------------------------------------
    shared_pool = [
        # Input 1: (e.g., Distance or Bearing)
        MFParamTrapezoidal(-180.0, -180.0, -45.0, 0.0),
        MFParamTriangular(-30.0, 0.0, 30.0),
        MFParamTrapezoidal(0.0, 45.0, 180.0, 180.0),
        # Input 2: (e.g., TTI)
        MFParamTrapezoidal(0.0, 0.0, 1.5, 3.0),
        MFParamTriangular(2.0, 4.0, 6.0),
        MFParamTrapezoidal(5.0, 8.0, 20.0, 20.0),
        # Input 3: (e.g., Ship Speed / Turn Rate / Ammo)
        MFParamTrapezoidal(-500.0, -500.0, -50.0, 0.0),
        MFParamTriangular(-30.0, 0.0, 30.0),
        MFParamTrapezoidal(0.0, 50.0, 500.0, 500.0),
    ]

    # -------------------------------------------------------------------------
    # Rule Base Definition: Legible Linear Equations
    # Layout format: RuleGene(antecedents=[(input_idx, pool_idx), ...])
    # -------------------------------------------------------------------------

    # Subsystem 1: Thrust (Inputs: Distance[0], TTI[1], Speed[2])
    thrust_rules = [
        # Rule 1: IF Threat is Critically Close, THEN brake hard
        RuleGene(
            antecedents=[(0, 0), (1, 3), (2, 8)],
            consequent_coefficients=[-0.5, 0.2, -1.2],
            consequent_offset=-200.0,
        ),
        # Rule 2: IF Threat is Distant/Safe, THEN accelerate ahead smoothly
        RuleGene(
            antecedents=[(0, 2), (1, 5), (2, 7)],
            consequent_coefficients=[0.2, 0.1, -0.4],
            consequent_offset=150.0,
        ),
    ]

    # Subsystem 2: Turn Rate (Inputs: Bearing[0], TTI[1], TurnRate[2])
    turn_rules = [
        # Rule 1: IF Target is to the Right, THEN rotate positive right
        RuleGene(
            antecedents=[(0, 2), (1, 4), (2, 7)],
            consequent_coefficients=[1.5, 0.0, -0.8],
            consequent_offset=45.0,
        ),
        # Rule 2: IF Target is to the Left, THEN rotate negative left
        RuleGene(
            antecedents=[(0, 0), (1, 4), (2, 7)],
            consequent_coefficients=[1.5, 0.0, -0.8],
            consequent_offset=-45.0,
        ),
    ]

    # Subsystem 3: Fire Trigger (Inputs: Bearing[0], Distance[1], AmmoRatio[2])
    fire_rules = [
        # Rule 1: IF Target is dead ahead AND Close AND Ammo plenty -> FIRE!
        RuleGene(
            antecedents=[(0, 1), (1, 0), (2, 8)],
            consequent_coefficients=[-0.1, -0.2, 0.5],
            consequent_offset=50.0,  # Positive value outputs > 0.0 -> True
        ),
        # Rule 2: IF Target is misaligned OR out of ammo -> HOLD FIRE
        RuleGene(
            antecedents=[(0, 2), (1, 2), (2, 6)],
            consequent_coefficients=[0.0, 0.0, 0.0],
            consequent_offset=-50.0,  # Negative value outputs <= 0.0 -> False
        ),
    ]

    return ControllerChromosome(
        thrust_blueprint=SubsystemChromosome(
            membership_function_pool=shared_pool,
            rules=thrust_rules,
        ),
        turn_blueprint=SubsystemChromosome(
            membership_function_pool=shared_pool,
            rules=turn_rules,
        ),
        fire_blueprint=SubsystemChromosome(
            membership_function_pool=shared_pool,
            rules=fire_rules,
        ),
    )


def _suppress_worker_interrupts() -> None:
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)


def main() -> None:
    dill.settings["recurse"] = True
    args = get_trainer_config()

    template = _make_structural_template()
    flat_genes_length = len(flatten_chromosome(template))
    weights = (1.0, 1.0, 1.0)

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
        algorithm = run_simple_evolution
        start_generation = 1
        rng = np.random.default_rng()
        logbook = Logbook()
        history_logger = HistoryLogger()
        hall_of_fame = HallOfFame(max_size=1)
        population = None

    initial_seed = int(rng.integers(1, 1000000))
    fitness_evaluator = FuzzyFitnessEvaluator(
        template=template,
        evaluation_seed=initial_seed,
    )

    individual_class = get_individual_class(weights)

    def fuzzy_individual_factory() -> Individual:
        raw_random_genes = [
            float(rng.uniform(-180.0, 500.0)) for _ in range(flat_genes_length)
        ]
        valid_genes = repair_geometry_invariants(raw_random_genes, template)
        return individual_class(valid_genes)

    mate_operator = cx_fuzzy_blend(template=template, rng=rng, alpha=0.3)
    mutate_operator = mut_fuzzy_gaussian(
        template=template,
        rng=rng,
        mu=0.0,
        sigma=0.1,
    )
    select_operator = sel_tournament(tournament_size=3)

    statistics = Statistics()
    toolbox = build_toolbox(
        evaluation_function=fitness_evaluator.evaluate_individual,
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
        fitness_evaluator.evaluation_seed = next_generation_seed

    print("Commencing evolutionary loop...")
    with ProcessPool(initializer=_suppress_worker_interrupts) as executor:
        toolbox.register("map", executor.map)

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
