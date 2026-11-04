from __future__ import annotations

from typing import TYPE_CHECKING, Final

from kesslergame import TrainerEnvironment

from asian_asteroid_destroyers.chromosomes import hydrate_chromosome_blueprint
from asian_asteroid_destroyers.controllers import FuzzyController
from asian_asteroid_destroyers.scenarios import load_scenario_suite

if TYPE_CHECKING:
    from pathlib import Path

    from asian_asteroid_destroyers.chromosomes import ControllerChromosome
    from asian_asteroid_destroyers.evokit import Individual

__all__ = ["FuzzyFitnessEvaluator"]


class FuzzyFitnessEvaluator:
    __slots__: tuple[str, ...] = (
        "_template",
        "_profile_path",
        "_evaluation_seed",
        "_environment",
    )
    _template: Final[ControllerChromosome]
    _profile_path: Final[Path]
    _evaluation_seed: int
    _environment: Final[TrainerEnvironment]

    def __init__(
        self,
        template: ControllerChromosome,
        profile_path: Path,
        evaluation_seed: int,
    ) -> None:
        self._template = template
        self._profile_path = profile_path
        self._evaluation_seed = evaluation_seed
        self._environment = TrainerEnvironment()

    @property
    def evaluation_seed(self) -> int:
        return self._evaluation_seed

    @evaluation_seed.setter
    def evaluation_seed(self, seed: int) -> None:
        self._evaluation_seed = seed

    def evaluate_individual(self, individual: Individual) -> tuple[float, ...]:
        chromosome = hydrate_chromosome_blueprint(
            individual[:],
            self._template,
        )

        scenarios_pool = load_scenario_suite(
            {
                "profile_path": self._profile_path,
                "seed": self._evaluation_seed,
                "is_deterministic": False,
            }
        )
        scenario_count = len(scenarios_pool)

        total_survival_time = 0.0
        total_hits = 0.0
        total_volatility = 0.0

        run_simulation = self._environment.run

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
