from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from kesslergame import KesslerGame

from asian_asteroid_destroyers.analysis import EvolutionMetrics
from asian_asteroid_destroyers.chromosomes import (
    get_default_controller_template,
    hydrate_chromosome_blueprint,
)
from asian_asteroid_destroyers.controllers import FuzzyController
from asian_asteroid_destroyers.infrastructure import get_evaluation_config
from asian_asteroid_destroyers.scenarios import load_scenario_suite

if TYPE_CHECKING:
    from kesslergame.score import Score


def report_metrics(
    scenario_name: str,
    score: Score,
    controller: FuzzyController,
) -> None:
    print("=========================================")
    print(f" PERFORMANCE METRICS: {scenario_name}")
    print("=========================================")
    print(f"  Simulated Runtime : {score.sim_time:.2f} seconds")

    teams = score.teams
    if teams:
        primary_team = teams[0]
        accuracy = primary_team.accuracy * 100
        status = "SURVIVED" if primary_team.lives_remaining > 0 else "CRASHED"

        print(f"  Asteroids Hit     : {primary_team.asteroids_hit}")
        print(f"  Shots Fired       : {primary_team.shots_fired}")
        print(f"  Accuracy Ratio    : {accuracy:.1f}%")
        print(f"  Deaths Sustained  : {primary_team.deaths}")
        print(f"  Mission Status    : \033[1;32m{status}\033[0m")
    else:
        print("  Team performance records unavailable.")

    print(f"  Mean Volatility   : {controller.mean_volatility:.4f}")
    print("=========================================\n")


def main() -> None:
    print("Initializing Elite Controller Evaluation Framework...")
    args = get_evaluation_config()

    metrics_extractor = EvolutionMetrics(
        logbook_path=args.logbook_path,
        history_path=args.history_path,
    )

    print("Resolving historical evolution structures...")
    best_raw_genome = metrics_extractor.extract_best_genome()

    template_blueprint = get_default_controller_template()
    chromosome = hydrate_chromosome_blueprint(
        flat_genes=best_raw_genome,
        template=template_blueprint,
    )

    scenarios = load_scenario_suite(
        {
            "profile_path": args.scenarios_path,
            "seed": 1337,
            "is_deterministic": True,
        }
    )

    game_environment = KesslerGame()

    print(f"Successfully loaded {len(scenarios)} testing profiles.")
    print("Booting visual graphics rendering environment...")

    for scenario in scenarios:
        controller = FuzzyController(chromosome=chromosome)
        print(f"Executing scenario: {scenario.name}")

        score, _ = game_environment.run(
            scenario=scenario,
            controllers=[controller],
        )

        report_metrics(
            scenario_name=scenario.name or "Unnamed Scenario",
            score=score,
            controller=controller,
        )

    print("All performance evaluation profiles completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation context interrupted by user. Exiting cleanly.")
        sys.exit(0)
