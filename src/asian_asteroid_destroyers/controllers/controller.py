# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.
from __future__ import annotations

from typing import Any, Dict, Tuple

state: Dict[str, Any]
state: Tuple[str, Any]

from typing import TYPE_CHECKING

import numpy as np
import skfuzzy as skf
import skfuzzy.control as ctrl
from kesslergame import KesslerController

if TYPE_CHECKING:
    from kesslergame.kessler_game import GameState, ShipState

# from controllers.ai.sa.sa import SA
from asian_asteroid_destroyers.controllers.sa.sa import SA
from asian_asteroid_destroyers.controllers.sa.util.helpers import trim_angle


class MyFuzzyController(KesslerController):
    def __init__(self, chromosome=None):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...
        self.chromosome = chromosome  # this leaves chromosome object as None - will trigger default behavior
        # or you can define your chromosome values (i.e. paste from your best found solution json) so your controller is
        # the same as the one found through your training.
        # self.chromosome = [
        #     0.2631645225918112,
        #     0.9352976507138837,
        #     0.9194881615834452,
        #     0.865825520047121,
        #     0.9537717256003297,
        #     0.3332987471941833,
        #     0.15500677821329611,
        #     0.6103279406781605,
        #     0.700110463994626,
        #     0.28645661550432655,
        #     0.7817669414563869,
        #     0.3242896160611243,
        #     0.2193792619771311,
        #     0.9187346087146049,
        #     0.7565990950849348,
        #     0.378982875637626,
        #     0.20469566850073984,
        #     0.4221919062502967,
        #     0.09472152646554266,
        #     0.22444174766372227,
        #     0.43972091554025095,
        #     0.3024938255921993,
        #     0.03951149080348482,
        #     0.07014695710862251,
        #     0.5085111197870378,
        #     0.7217226903052553,
        #     0.8478160198517868,
        #     0.9622056814666377,
        #     0.9292672901679256,
        #     0.8763434435737334,
        #     0.8715824359186305,
        #     0.622211987772582,
        #     0.09103610636756931,
        #     0.42932650446832776,
        #     0.11151119794790532,
        #     0.6449315965771553,
        #     0.39484371578702315,
        #     0.5792033115832056,
        #     0.31492571381606194,
        #     0.6833570548284794,
        #     0.8625524374751714,
        #     0.8127256586152101,
        #     0.2550469991459845,
        #     0.19763655967886684,
        #     0.8740087388142191,
        #     0.9902384288505531,
        #     0.8321982213704795,
        #     0.6787979639950257,
        #     0.3945538617015545,
        #     0.8955970115205919
        #   ]
        self.aiming_fis = None
        self.aiming_fis_sim = None
        self.moving_fis = None
        self.moving_fis_sim = None
        self.normalization_dist = None
        self.sa = SA()

        # I put this in a separate function for cleanliness in the init procedure, but this just calls the functions
        # that create your FIS functions
        self.create_fuzzy_systems()

    def actions(
        self,
        ship_state: ShipState,
        game_state: GameState,
    ) -> tuple[float, float, bool, bool]:
        self.update_situational_awareness(ship_state, game_state)
        asteroid = self.get_nearest_asteroid()
        angle, distance = self.calculate_features(asteroid)

        aim_output, thrust_output = self.run_fuzzy_systems(angle, distance)

        turn_rate = self.calculate_turn_rate(
            ship_state,
            trim_angle(asteroid.bearing - self.sa.ownship.heading),
            aim_output,
        )

        thrust = thrust_output * 480.0
        return (thrust, turn_rate, True, False)

    def calculate_turn_rate(self, ship_state, relative_angle, aim_output):
        desired_angle = aim_output * 190
        difference = trim_angle(desired_angle - relative_angle)
        if difference < -0.5:
            return ship_state["turn_rate_range"][1]
        if difference > 0.5:
            return ship_state["turn_rate_range"][0]
        return 0

    def convert_rule_values(self, rule_values):
        bins = np.array([1 / 3, 2 / 3])
        indices = np.digitize(rule_values, bins)
        return [int(min(max(index, 0), 2)) for index in indices]

    def calculate_features(self, asteroid):
        relative_angle = trim_angle(asteroid.bearing - self.sa.ownship.heading)
        normalized_angle = self.sa.norm_angle(relative_angle)
        normalized_distance = self.sa.norm_distance(asteroid.distance)
        return normalized_angle, normalized_distance

    def create_rules_from_outputs(self, distance, angle, output, output_names):
        distance_mfs = [distance["close"], distance["medium"], distance["far"]]
        angle_mfs = [angle["negative"], angle["zero"], angle["positive"]]
        output_mfs = {
            "negative": output["negative"],
            "zero": output["zero"],
            "positive": output["positive"],
        }
        rules = []
        count = 0
        for angle_mf in angle_mfs:
            for distance_mf in distance_mfs:
                rules.append(
                    ctrl.Rule(
                        distance_mf & angle_mf, output_mfs[output_names[count]]
                    )
                )
                count += 1
        return rules

    def run_fuzzy_systems(self, angle, distance):
        aiming_output = self.tsk_output(
            angle,
            distance,
            self.aiming_centers,
            self.aiming_consequents,
            self.aiming_scale,
        )

        thrust_output = self.tsk_output(
            angle,
            distance,
            self.moving_centers,
            self.moving_components,
            self.moving_scale,
        )

        return aiming_output, thrust_output

    def setup_chromosome_memberships(
        self,
        distance,
        angle,
        output,
        distance_index,
        angle_index,
        output_index,
    ):
        if self.chromosome is None:
            raise ValueError("Chromosome must be initialized")
        distance_value = float(
            np.clip(self.chromosome[distance_index], 0.01, 0.99)
        )
        angle_value = float(
            np.clip(self.chromosome[angle_index] * 2 - 1, -0.99, 0.99)
        )
        output_value = float(
            np.clip(self.chromosome[output_index] * 2 - 1, -0.99, 0.99)
        )
        # output = self.chromosome[output_index]

        distance["close"] = skf.trimf(
            distance.universe, [0.0, 0.0, distance_value]
        )
        distance["medium"] = skf.trimf(
            distance.universe, [0.0, distance_value, 1.0]
        )
        distance["far"] = skf.trimf(
            distance.universe, [distance_value, 1.0, 1.0]
        )

        # self.create_triangular_memberships(distance, distance_value)
        self.create_triangular_memberships(angle, angle_value)
        self.create_triangular_memberships(output, output_value)

    def create_chromosome_rules(self, distance, angle, output, rule_start=3):
        if self.chromosome is None:
            raise ValueError("Chromosome must be initialized")
        distance_mfs = [distance["close"], distance["medium"], distance["far"]]
        angle_mfs = [angle["negative"], angle["zero"], angle["positive"]]
        output_mfs = [output["negative"], output["zero"], output["positive"]]

        rule_values = self.chromosome[rule_start : rule_start + 9]
        indices = self.convert_rule_values(rule_values)

        rules = []
        count = 0
        for angle_mf in angle_mfs:
            for distance_mf in distance_mfs:
                rules.append(
                    ctrl.Rule(
                        distance_mf & angle_mf, output_mfs[indices[count]]
                    )
                )
                count += 1
        return rules

    def create_output_memberships(self, output):
        output["negative"] = skf.trimf(output.universe, [-1.0, -1.0, 0.0])
        output["zero"] = skf.trimf(output.universe, [-1.0, 0.0, 1.0])
        output["positive"] = skf.trimf(output.universe, [0.0, 1.0, 1.0])

    def create_triangular_memberships(self, variable, center: float):
        variable["negative"] = skf.trimf(
            variable.universe, [-1.0, -1.0, center]
        )
        variable["zero"] = skf.trimf(variable.universe, [-1.0, center, 1.0])
        variable["positive"] = skf.trimf(variable.universe, [center, 1.0, 1.0])

    def setup_default_memberships(self, distance, angle, output):
        distance.automf(3, names=["close", "medium", "far"])
        angle.automf(3, names=["negative", "zero", "positive"])
        self.create_output_memberships(output)

    def create_default_rules(self, distance, angle, output):
        output_names = [
            "negative, negative",
            "negative",
            "zero, zero",
            "zero",
            "positive, positive",
            "positive",
        ]

        return self.create_rules_from_outputs(
            distance, angle, output, output_names
        )

    def create_fuzzy_variables(self, output_name: str):
        distance = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "distance")
        angle = ctrl.Antecedent(np.linspace(-1.0, 1.0, 11), "angle")
        output = ctrl.Consequent(np.linspace(-1.0, 1.0, 11), output_name)
        return [distance, angle, output]

    def create_fuzzy_systems(self):
        self.create_aiming_fis()
        self.create_moving_fis()

    def create_aiming_fis(self):
        if self.chromosome is None:
            self.aiming_centers = (0.5, 0.0)
            self.aiming_scale = 1.0
            self.aiming_consequents = [
                -1.0,
                -0.5,
                0.0,
                -0.5,
                0.0,
                0.5,
                0.0,
                0.5,
                1.0,
            ]
        else:
            self.aiming_centers = (
                float(np.clip(self.chromosome[0], 0.01, 0.09)),
                float(np.clip(self.chromosome[1] * 2 - 1, 0.99, 0.99)),
            )
            self.aiming_scale = float(np.clip(self.chromosome[2], 0.01, 1.0))
            self.aiming_consequents = [
                float(np.clip(gene * 2 - 1, -1.0, 1.0))
                for gene in self.aiming_consequents
            ]

    def update_situational_awareness(self, ship_state, game_state):
        self.sa.update(ship_state, game_state)

    def get_nearest_asteroid(self):
        return self.sa.ownship.nearest_n(1)[0]

    def create_moving_fis(self):
        if self.chromosome is None:
            self.moving_centers = (0.5, 0.0)
            self.moving_scale = 1.0
            self.moving_consequents = [
                -1.0,
                -0.5,
                0.0,
                -0.5,
                0.0,
                0.5,
                0.0,
                0.5,
                1.0,
            ]
        else:
            self.moving_centers = (
                float(np.clip(self.chromosome[12], 0.01, 0.99)),
                float(np.clip(self.chromosome[13] * 2 - 1, -0.99, 0.99)),
            )
            self.moving_scale = float(np.clip(self.chromosome[14], 0.01, 1.0))
            self.moving_consequents = [
                float(np.clip(gene * 2 - 1, -1.0, 1.0))
                for gene in self.chromosome[15:24]
            ]

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Fuzzy Test1"

    # @property
    # def custom_sprite_path(self) -> str:
    #     return "Neo.png"
    def get_memberships(self, value, variable_type, center):
        if variable_type == "distance":
            triangles = [
                [0.0, 0.0, center],
                [0.0, center, 0.0],
                [center, 0.0, 0.0],
            ]
        else:
            triangles = [
                [-1.0, -1.0, center],
                [-1.0, center, 1.0],
                [center, 1.0, 1.0],
            ]
        return [
            float(skf.trimf(np.array([value]), triangle)[0])
            for triangle in triangles
        ]

    def tsk_output(self, angle, distance, output, centers, consequents, scale):
        angle_memberships = self.get_memberships(angle, "angle", centers[1])
        distance_memberships = self.get_memberships(
            distance, "distance", centers[0]
        )
        weighted_sum = 0.0
        activation_sum = 0.0
        rule_index = 0
        for angle_index in range(3):
            for distance_index in range(3):
                activation = (
                    angle_memberships[angle_index]
                    * distance_memberships[distance_index]
                )
                consequent = scale * consequents[rule_index]
                weighted_sum += activation * consequent
                activation_sum += activation
                rule_index += 1
        if activation_sum == 0:
            return 0.0
        return weighted_sum / activation_sum
