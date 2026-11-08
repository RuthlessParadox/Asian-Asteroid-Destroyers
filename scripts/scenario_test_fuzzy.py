# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import sys
import time

import numpy as np
from kesslergame.settings_dicts import SettingsDict

sys.path.append("")

from kesslergame import GraphicsType, KesslerController, KesslerGame, Scenario

from asian_asteroid_destroyers.chromosomes import (
    ControllerChromosome,
    MFParamTriangular,
    RuleGene,
    SubsystemChromosome,
    flatten_chromosome,
    get_default_controller_template,
    hydrate_chromosome_blueprint,
)
from asian_asteroid_destroyers.controllers.fuzzy_controller import (
    FuzzyController,
)


def create_subsystem_template(
    input_ranges: list[tuple[float, float]],
    output_default: float = 0.0,
) -> SubsystemChromosome:
    membership_functions = []
    for (
        minimum,
        maximum,
    ) in input_ranges:
        middle = (minimum + maximum) / 2
        width = maximum - minimum

        membership_functions.extend(
            [
                MFParamTriangular(minimum, minimum, middle),
                MFParamTriangular(minimum, middle, maximum),
                MFParamTriangular(middle, maximum, minimum),
            ]
        )
    rules = [
        RuleGene(
            antecedents=[
                (0, 1),
                (1, 4),
                (2, 7),
            ],
            consequent_coefficients=[0.0, 0.0, 0.0],
            consequent_offset=output_default,
            activation_gate=0.0,
        )
    ]
    return SubsystemChromosome(
        membership_function_pool=membership_functions,
        rules=rules,
    )


chromosome_template = get_default_controller_template()

flat_genes = flatten_chromosome(chromosome_template)
chromosome = hydrate_chromosome_blueprint(
    flat_genes=flat_genes,
    template=chromosome_template,
)

# def evaluate(individual):
#    chromosome: ControllerChromosome = hydrate_chromosome_blueprint(
#        flat_genes=individual,
#        template=chromosome_template,
#    )
#    controller = FuzzyController(chromosome=chromosome)

# Define game scenario
# my_test_scenario = Scenario(name='Test Scenario',
#                             num_asteroids=10,
#                             ship_states=[
#                                 {'position': (400, 400), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 3},
#                                 # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
#                             ],
#                             map_size=(1000, 800),
#                             time_limit=60,
#                             ammo_limit_multiplier=0,
#                             stop_if_no_ammo=False)

my_test_scenario = Scenario(
    name="Test Scenario",
    asteroid_states=[
        {"position": (200, 400), "angle": 0, "speed": 10, "size": 4}
    ],
    ship_states=[
        {
            "position": (400, 400),
            "angle": 90,
            "lives": 3,
            "team": 1,
            "mines_remaining": 3,
        },
        # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False,
)

# Define Game Settings
game_settings: SettingsDict = {
    "perf_tracker": True,
    "graphics_type": GraphicsType.Tkinter,
    "realtime_multiplier": 1,
    "graphics_obj": None,
    "frequency": 60,
}

game = KesslerGame(
    settings=game_settings
)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game


range = np.random.default_rng(42)
pre = time.perf_counter()
# score, perf_data = (
# game.run(
# scenario=my_test_scenario,
#    controllers=[
#        FuzzyController(
#            [
#                0.5639007014925158,
#                0.8201832063971009,
#                0.5800650926910023,
#                0.45551613783025213,
#                0.39329422608792985,
#                0.9017773815234136,
#                0.8325337413417127,
#                0.5921424691280988,
#                0.6241952154355225,
#                0.3001290246334073,
#                0.11994134811268875,
#                0.13549991931863076,
#                0.28989468783309724,
#                0.3806332253773407,
#                0.14293660780706718,
#                0.6612293964867574,
#                0.3157307369718849,
#                0.7594206031605566,
#                0.02759597581613693,
#                0.11292744595505078,
#                0.4585235349212474,
#                0.7232957272238553,
#                0.33262739409019604,
#                0.054117230232812674,
#                0.9444214813585389,
#                0.4972704564185404,
#                0.3394477485603036,
#                0.5340822685070491,
#                0.4022210607694814,
#                0.07261816477007321,
#                0.9448454990905524,
#                0.2693323762461447,
#                0.23614916310455702,
#                0.41335495304112935,
#                0.8355630123007904,
#                0.13099441418805302,
#                0.4519241227287687,
#                0.6239811307090696,
#                0.7353533746893613,
#                0.8774493497139241,
#                0.7137812912512537,
#                0.05330133773971035,
#                0.6385849116218539,
#                0.883746629970664,
#                0.3410954601299995,
#                0.7474426858001362,
#                0.3123208438673374,
#                0.9394852290833192,
#                0.4482481624016552,
#                0.7047415494275244,
#            ]
#    ),
# )
#        FuzzyController(
#            [
#                0.5639007014925158,
#                0.8201832063971009,
#                0.5800650926910023,
#                0.45551613783025213,
#                0.39329422608792985,
#                0.9017773815234136,
#                0.8325337413417127,
#                0.5921424691280988,
#                0.6241952154355225,
#                0.3001290246334073,
#                0.11994134811268875,
#                0.13549991931863076,
#                0.28989468783309724,
#                0.3806332253773407,
#                0.14293660780706718,
#                0.6612293964867574,
#                0.3157307369718849,
#                0.7594206031605566,
#                0.02759597581613693,
#                0.11292744595505078,
#                0.4585235349212474,
#                0.7232957272238553,
#                0.33262739409019604,
#                0.054117230232812674,
#                0.9444214813585389,
#                0.4972704564185404,
#                0.3394477485603036,
#                0.5340822685070491,
#                0.4022210607694814,
#                0.07261816477007321,
#                0.9448454990905524,
#                0.2693323762461447,
#                0.23614916310455702,
#                0.41335495304112935,
#                0.8355630123007904,
#                0.13099441418805302,
#                0.4519241227287687,
#                0.6239811307090696,
#                0.7353533746893613,
#                0.8774493497139241,
#                0.7137812912512537,
#                0.05330133773971035,
#                0.6385849116218539,
#                0.883746629970664,
#                0.3410954601299995,
#                0.7474426858001362,
#                0.3123208438673374,
#                0.9394852290833192,
#                0.4482481624016552,
#                0.7047415494275244,
#            ]
#        ),
#    ],
# )

controller = FuzzyController(chromosome=chromosome)

controllers: list[KesslerController] = [controller]
score, perf_data = game.run(
    scenario=my_test_scenario,
    controllers=controllers,
)
# Print out some general info about the result
print("Scenario eval time: " + str(time.perf_counter() - pre))
print(score.stop_reason)
print("Asteroids hit: " + str([team.asteroids_hit for team in score.teams]))
print("Deaths: " + str([team.deaths for team in score.teams]))
print("Accuracy: " + str([team.accuracy for team in score.teams]))
print("Mean eval time: " + str([team.mean_eval_time for team in score.teams]))
