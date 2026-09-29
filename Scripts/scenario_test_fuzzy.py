# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import sys

sys.path.append('.')

from kesslergame import Scenario, KesslerGame, GraphicsType
from Scenarios.example_scenarios import *
from MyAIController.example_controller_fuzzy import MyFuzzyController
from MyAIController.example_controller_fuzzy2 import MyFuzzyController2

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

my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=5,
                            seed=3,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 60}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[MyFuzzyController([
    0.7667558104926862,
    0.3442682637965213,
    0.49396298588103826,
    0.4864774798796745,
    0.8748314903850923,
    0.02145503563122131,
    0.47272168137785997,
    0.5533701211924614,
    0.05770887632342983,
    0.1060388316597417,
    0.577028900104021,
    0.585291741691332,
    0.7567710730178083,
    0.22790752402860578,
    0.16239390514047247,
    0.6195965767071961,
    0.24388540603571296,
    0.41071904255765046,
    0.7744293730056345,
    0.8503574099565501,
    0.943686811643859,
    0.8178764908072563,
    0.9966992085523995,
    0.2279217168657799,
    0.1373479165192697,
    0.18265114425674134,
    0.6611975644806934,
    0.8632446159388327,
    0.9501130482189761,
    0.38309490295987625,
    0.1003524057902716,
    0.019446849244120656,
    0.13304516779728093,
    0.7445557865285479,
    0.15663836699111577,
    0.819379686919093,
    0.29074961192649673,
    0.5300464426161686,
    0.05221697712951279,
    0.4476164510571894,
    0.6563623646476601,
    0.46382373183116876,
    0.39092040069918044,
    0.2578673904900568,
    0.20626533318550289,
    0.32576173205585746,
    0.9241702250842517,
    0.9259802140590035,
    0.9820311565496787,
    0.3303854980578882
  ]), MyFuzzyController([
    0.7667558104926862,
    0.3442682637965213,
    0.49396298588103826,
    0.4864774798796745,
    0.8748314903850923,
    0.02145503563122131,
    0.47272168137785997,
    0.5533701211924614,
    0.05770887632342983,
    0.1060388316597417,
    0.577028900104021,
    0.585291741691332,
    0.7567710730178083,
    0.22790752402860578,
    0.16239390514047247,
    0.6195965767071961,
    0.24388540603571296,
    0.41071904255765046,
    0.7744293730056345,
    0.8503574099565501,
    0.943686811643859,
    0.8178764908072563,
    0.9966992085523995,
    0.2279217168657799,
    0.1373479165192697,
    0.18265114425674134,
    0.6611975644806934,
    0.8632446159388327,
    0.9501130482189761,
    0.38309490295987625,
    0.1003524057902716,
    0.019446849244120656,
    0.13304516779728093,
    0.7445557865285479,
    0.15663836699111577,
    0.819379686919093,
    0.29074961192649673,
    0.5300464426161686,
    0.05221697712951279,
    0.4476164510571894,
    0.6563623646476601,
    0.46382373183116876,
    0.39092040069918044,
    0.2578673904900568,
    0.20626533318550289,
    0.32576173205585746,
    0.9241702250842517,
    0.9259802140590035,
    0.9820311565496787,
    0.3303854980578882
  ])])
# score, perf_data = game.run(scenario=my_test_scenario, controllers=[MyFuzzyController2(), MyFuzzyController()])

# Print out some general info about the result
print('Scenario eval time: '+str(time.perf_counter()-pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
