# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

from kesslergame import Scenario


# Small game scenario with 5 random (seeded) asteroid initial conditions
scenario1 = Scenario(name='Scenario 1',
                            num_asteroids=5,
                            seed=3,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=30,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)


# Slightly bigger game scenario with 10 random (seeded) asteroid initial conditions
scenario2 = Scenario(name='Scenario 2',
                            num_asteroids=10,
                            seed=3,
                            ship_states=[
                                {'position': (250, 250), 'angle': 180, 'lives': 1, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=30,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

scenario3 = Scenario(name='Scenario 3',
                            num_asteroids=10,
                            seed=3,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=30,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

scenario4 = Scenario(name='Scenario 4',
                            num_asteroids=10,
                            seed=3,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=30,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

scenario5 = Scenario(name='Scenario 5',
                            num_asteroids=10,
                            seed=3,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=30,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)
training_set = [scenario1,
                scenario2,
                scenario3,
                scenario4,
                scenario5]
