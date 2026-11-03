from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from kesslergame import TrainerEnvironment

from asian_asteroid_destroyers.chromosomes import hydrate_chromosome_blueprint
from asian_asteroid_destroyers.controllers import FuzzyController
from asian_asteroid_destroyers.scenarios import create_training_portfolio

if TYPE_CHECKING:
    from asian_asteroid_destroyers.chromosomes import ControllerChromosome
    from asian_asteroid_destroyers.evokit import FitnessEvaluator, Individual

__all__ = ["FuzzyFitnessEvaluator", "get_evaluator_callback"]


class FuzzyFitnessEvaluator:
    __slots__: tuple[str, ...] = (
        "_template",
        "_evaluation_seed",
        "_environment",
    )
    _template: Final[ControllerChromosome]
    _evaluation_seed: Final[int]
    _environment: Final[TrainerEnvironment]

    def __init__(
        self,
        template: ControllerChromosome,
        evaluation_seed: int,
    ) -> None:
        self._template = template
        self._evaluation_seed = evaluation_seed
        self._environment = TrainerEnvironment()

    def evaluate_individual(self, individual: Individual) -> tuple[float, ...]:
        chromosome = hydrate_chromosome_blueprint(
            individual[:],
            self._template,
        )

        local_rng = np.random.default_rng(self._evaluation_seed)
        scenarios_pool = create_training_portfolio(local_rng)
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


def get_evaluator_callback(
    template: ControllerChromosome,
    seed: int,
) -> FitnessEvaluator:
    evaluator = FuzzyFitnessEvaluator(template=template, evaluation_seed=seed)
    return evaluator.evaluate_individual
