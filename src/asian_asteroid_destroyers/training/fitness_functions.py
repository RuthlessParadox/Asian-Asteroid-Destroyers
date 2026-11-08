# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import random
import sys
import time

sys.path.append("")

import random

import numpy as np
from kesslergame import GraphicsType, KesslerGame
from kesslergame.kessler_game import SettingsDict

from asian_asteroid_destroyers.controllers.fuzzy_controller import (
    FuzzyController,
)
from asian_asteroid_destroyers.scenarios import (
    create_training_portfolio,
)

DEFAULT_WEIGHTS = {
    "fraction_total_asteroids_hit": 1.0,
    "accuracy": 1.0,
    "deaths": -0.2,
}


def calculate_fitness(team, weights):
    score = 0.0
    score += (
        weights.get("fraction_total_asteroids_hit", 0.0)
        * team.fraction_total_asteroids_hit
    )
    score += weights.get("accuracy", 0.0) * team.accuracy
    score += weights.get("deaths", 0.0) * team.deaths

    return score


def example_fitness(individual, weights=None, settings=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS
    controller = FuzzyController(chromosome=individual)
    total_score = 0
    game_settings: SettingsDict = {
        "perf_tracker": False,
        "graphics_type": GraphicsType.NoGraphics,
        "realtime_multiplier": 0,
        "graphics_obj": None,
        "frequency": 30,
        "time_limit": 30,
        "competition_safe_mode": False,
    }
    if settings is not None:
        game_settings.update(settings)
    game = KesslerGame(settings=game_settings)  # Instantiate the game object
    # game = TrainerEnvironment(settings=game_settings)  # Kessler default settings for training - has some defaults that override what we want though

    range = np.random.default_rng(42)
    for index, scenario in enumerate(create_training_portfolio(range)):
        start = time.perf_counter()
        print(f"Starting scenario {index + 1}", flush=True)

        result, _ = game.run(scenario=scenario, controllers=[controller])
        print(
            f"Scenario {index + 1} completed in "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )
        score = (
            result.teams[0].fraction_total_asteroids_hit
            + result.teams[0].accuracy
        )
        team = result.teams[0]
        total_score += calculate_fitness(team, weights)

    return (total_score,)


def create_fitness_function(weights=None):
    selected_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

    def fitness(individual):
        return example_fitness(
            individual,
            weights=selected_weights,
        )

    return fitness


if __name__ == "__main__":
    test_individual = [random.random() for _ in range(24)]
    print(test_individual)

    test_score = example_fitness(individual=test_individual)
    print(test_score)

    # print(score.stop_reason)
    # print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    # print('Deaths: ' + str([team.deaths for team in score.teams]))
    # print('Accuracy: ' + str([team.accuracy for team in score.teams]))
    # print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    # # test_controller = FuzzyController(individual)
    # print(test_controller)

    fitness = create_fitness_function(
        {
            "fraction_total_asteroids_hit": 2.0,
            "accuracy": 1.0,
            "deaths": -0.5,
        }
    )
