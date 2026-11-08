from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

from kesslergame import TrainerEnvironment

from asian_asteroid_destroyers.chromosomes import hydrate_chromosome_blueprint
from asian_asteroid_destroyers.controllers import FuzzyController
from asian_asteroid_destroyers.scenarios import (
    RootJsonSchema,
    load_scenario_suite,
)

if TYPE_CHECKING:
    from pathlib import Path

    from asian_asteroid_destroyers.chromosomes import ControllerChromosome
    from asian_asteroid_destroyers.evokit import Individual

__all__ = ["FuzzyFitnessEvaluator"]


class FuzzyFitnessEvaluator:
    __slots__: tuple[str, ...] = (
        "_template",
        "_cached_json_data",
        "_evaluation_seed",
        "_environment",
    )
    _template: Final[ControllerChromosome]
    _cached_json_data: Final[RootJsonSchema]
    _evaluation_seed: int
    _environment: Final[TrainerEnvironment]

    def __init__(
        self,
        template: ControllerChromosome,
        profile_path: Path,
        evaluation_seed: int,
    ) -> None:
        self._template = template
        self._evaluation_seed = evaluation_seed
        self._environment = TrainerEnvironment()

        if not profile_path.is_file():
            raise FileNotFoundError(
                f"Scenario profile sheet missing: {profile_path}"
            )

        with open(profile_path, "r", encoding="utf-8") as file:
            self._cached_json_data = cast(RootJsonSchema, json.load(file))

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
                "json_data": self._cached_json_data,
                "seed": self._evaluation_seed,
                "is_deterministic": False,
            }
        )
        scenario_count = len(scenarios_pool)

        total_asteroid_fraction = 0.0
        total_accuracy = 0.0
        total_survival_fraction = 0.0
        total_deaths = 0.0
        total_volatility = 0.0

        run_simulation = self._environment.run

        for scenario in scenarios_pool:
            controller = FuzzyController(chromosome)

            try:
                score_object, _ = run_simulation(
                    scenario=scenario,
                    controllers=[controller],
                )

                time_limit = (
                    scenario.time_limit if scenario.time_limit > 0.0 else 1.0
                )
                total_survival_fraction += score_object.sim_time / time_limit
                total_volatility += controller.mean_volatility

                for team in score_object.teams:
                    total_asteroid_fraction += (
                        team.fraction_total_asteroids_hit
                    )
                    total_accuracy += team.accuracy
                    total_deaths += team.deaths
            except AssertionError:
                total_deaths += 10.0
                total_volatility += 10.0
                continue

        return (
            total_asteroid_fraction / scenario_count,
            total_accuracy / scenario_count,
            total_survival_fraction / scenario_count,
            total_deaths / scenario_count,
            total_volatility / scenario_count,
        )
