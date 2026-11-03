from __future__ import annotations

from typing import TYPE_CHECKING, Final

from kesslergame import TrainerEnvironment

from asian_asteroid_destroyers.chromosomes import hydrate_chromosome_blueprint
from asian_asteroid_destroyers.controllers import FuzzyController

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kesslergame import Scenario

    from asian_asteroid_destroyers.chromosomes import ControllerChromosome
    from asian_asteroid_destroyers.evokit import Individual

__all__ = ["FuzzyFitnessEvaluator"]


class FuzzyFitnessEvaluator:
    __slots__: tuple[str, ...] = ("_template", "_scenarios", "_environment")
    _template: Final[ControllerChromosome]
    _scenarios: Final[Sequence[Scenario]]
    _environment: Final[TrainerEnvironment]

    def __init__(
        self,
        template: ControllerChromosome,
        scenarios: Sequence[Scenario],
    ) -> None:
        if not scenarios:
            raise ValueError("Evaluation requires at least one scenario.")
        self._template = template
        self._scenarios = scenarios
        self._environment = TrainerEnvironment()

    def evaluate_individual(self, individual: Individual) -> tuple[float, ...]:
        chromosome = hydrate_chromosome_blueprint(
            individual[:],
            self._template,
        )

        total_survival_time = 0.0
        total_hits = 0.0
        total_volatility = 0.0

        run_simulation = self._environment.run
        scenarios_pool = self._scenarios
        scenario_count = len(scenarios_pool)

        for scenario in scenarios_pool:
            controller = FuzzyController(chromosome)

            score_object, _ = run_simulation(
                scenario=scenario,
                controllers=[controller],
            )

            total_survival_time += float(score_object.sim_time)
            total_volatility += controller.mean_volatility

            for team in score_object.teams:
                total_hits += float(team.asteroids_hit)

        return (
            total_survival_time / scenario_count,
            total_hits / scenario_count,
            -(total_volatility / scenario_count),
        )
