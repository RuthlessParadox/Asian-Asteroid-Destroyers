from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import numpy as np
from kesslergame import KesslerController
from typing_extensions import override

from asian_asteroid_destroyers.chromosomes import compile_subsystem
from asian_asteroid_destroyers.geometry import (
    denorm_thrust,
    denorm_turn_rate,
    norm_angle,
    norm_speed_ship,
    norm_tti,
    norm_turn_rate_ship,
    norm_wrapped_distance,
)
from asian_asteroid_destroyers.sa import SituationalAwareness

if TYPE_CHECKING:
    from kesslergame.state_models import GameState, ShipState

    from asian_asteroid_destroyers.chromosomes import ControllerChromosome
    from asian_asteroid_destroyers.fuzzy import TSKInferenceEngine
    from asian_asteroid_destroyers.sa import AsteroidView


class _TacticalFeatureExtractor:
    __slots__: tuple[str, ...] = ("_sa",)
    _sa: Final[SituationalAwareness]

    def __init__(self, sa: SituationalAwareness) -> None:
        self._sa = sa

    def get_predictive_bearing(self, asteroid: AsteroidView) -> float:
        ship = self._sa.agent_ship
        tti = asteroid.tti

        if tti is None or np.isnan(tti):
            return asteroid.bearing_wrap

        ast_vel = np.array(asteroid.velocity, dtype=np.float64)
        ship_vel = np.array(ship.velocity, dtype=np.float64)
        rel_vel = ast_vel - ship_vel

        ast_pos = np.array(asteroid.position_wrap, dtype=np.float64)
        predictive_pos = ast_pos + (rel_vel * tti)

        map_size = self._sa.map_size
        wrapped_proj = np.mod(predictive_pos, map_size)

        delta = wrapped_proj - np.array(ship.position, dtype=np.float64)

        if np.allclose(delta, 0.0) or np.any(np.isnan(delta)):
            return float(ship.heading)

        return cast(
            float,
            np.degrees(
                cast(
                    np.float64,
                    np.arctan2(
                        cast(np.float64, delta[1]),
                        cast(np.float64, delta[0]),
                    ),
                )
            ),
        )

    def get_offensive_inputs(self) -> list[float]:
        """Features for tracking, aligning with, and chasing down targets."""
        nearest = self._sa.agent_ship.nearest_n_asteroids_wrapped(1)
        ship = self._sa.agent_ship

        if not nearest:
            return [0.0, 1.0, norm_turn_rate_ship(ship.turn_rate)]

        target = nearest[0]
        pred_bearing = self.get_predictive_bearing(target)
        angular_error = pred_bearing - ship.heading
        normalized_error = norm_angle((angular_error + 180.0) % 360.0 - 180.0)

        norm_dist = norm_wrapped_distance(
            target.distance_wrap,
            self._sa.map_width,
            self._sa.map_height,
        )

        return [
            normalized_error,
            norm_dist,
            norm_turn_rate_ship(ship.turn_rate),
        ]

    def get_defensive_inputs(self) -> list[float]:
        """Features for evasive micro-maneuvering and wall avoidance."""
        soonest = self._sa.agent_ship.soonest_impact_n_asteroids(1)
        ship = self._sa.agent_ship

        if not soonest:
            return [0.0, 1.0, norm_speed_ship(ship.speed)]

        threat = soonest[0]
        tti = threat.tti
        normalized_tti = norm_tti(tti) if tti is not None else 1.0

        # Calculate side-slip vector relative to threat velocity vector
        ast_vel = np.array(threat.velocity, dtype=np.float64)
        ship_vel = np.array(ship.velocity, dtype=np.float64)
        rel_vel = ast_vel - ship_vel

        delta_p = np.array(threat.position_wrap, dtype=np.float64) - np.array(
            ship.position,
            dtype=np.float64,
        )

        dist = cast(float, np.maximum(np.linalg.norm(delta_p), 1.0))

        # Cross product evaluates if threat is tracking center-mass or passing
        cross_prod = float(np.cross(rel_vel, delta_p) / dist)
        norm_lateral_threat = max(-1.0, min(cross_prod / 500.0, 1.0))

        return [
            norm_lateral_threat,
            normalized_tti,
            norm_speed_ship(ship.speed),
        ]

    def get_gate_inputs(self) -> list[float]:
        """Macro-environmental context tracking cluster densities."""
        soonest = self._sa.agent_ship.soonest_impact_n_asteroids(1)

        if not soonest:
            return [1.0, 0.0]

        threat = soonest[0]
        tti = threat.tti
        normalized_tti = norm_tti(tti) if tti is not None else 1.0

        # Capture density within immediate tactical radius (e.g., 300 units)
        surrounding = self._sa.agent_ship.nearest_n_asteroids_wrapped(8)
        dense_count = 0
        for ast in surrounding:
            if ast.distance_wrap < 300.0:
                dense_count += 1

        norm_density = min(float(dense_count) / 8.0, 1.0)

        return [
            normalized_tti,
            norm_density,
        ]


class FuzzyController(KesslerController):
    __slots__: tuple[str, ...] = (
        "_sa",
        "_extractor",
        "_thrust_offensive_engine",
        "_thrust_defensive_engine",
        "_turn_offensive_engine",
        "_turn_defensive_engine",
        "_threat_gate_engine",
        "_emergency_tti_threshold",
        "_damping_factor",
        "_last_thrust",
        "_last_turn",
        "_total_accumulated_volatility",
        "_processed_steps",
    )

    _sa: Final[SituationalAwareness]
    _extractor: Final[_TacticalFeatureExtractor]

    _thrust_offensive_engine: Final[TSKInferenceEngine]
    _thrust_defensive_engine: Final[TSKInferenceEngine]
    _turn_offensive_engine: Final[TSKInferenceEngine]
    _turn_defensive_engine: Final[TSKInferenceEngine]
    _threat_gate_engine: Final[TSKInferenceEngine]

    _emergency_tti_threshold: Final[float]
    _damping_factor: Final[float]

    _last_thrust: float | None
    _last_turn: float | None
    _total_accumulated_volatility: float
    _processed_steps: int

    def __init__(self, chromosome: ControllerChromosome) -> None:
        super().__init__()

        self._sa = SituationalAwareness()
        self._extractor = _TacticalFeatureExtractor(self._sa)

        self._thrust_offensive_engine = compile_subsystem(
            chromosome.thrust_offensive
        )
        self._thrust_defensive_engine = compile_subsystem(
            chromosome.thrust_defensive
        )
        self._turn_offensive_engine = compile_subsystem(
            chromosome.turn_offensive
        )
        self._turn_defensive_engine = compile_subsystem(
            chromosome.turn_defensive
        )
        self._threat_gate_engine = compile_subsystem(chromosome.threat_gate)

        self._emergency_tti_threshold = 0.45
        self._damping_factor = 0.15

        self._last_thrust = None
        self._last_turn = None
        self._total_accumulated_volatility = 0.0
        self._processed_steps = 0

    @override
    def actions(
        self,
        ship_state: ShipState,
        game_state: GameState,
    ) -> tuple[float, float, bool, bool]:
        self._sa.update(ship_state, game_state)
        if self._sa.asteroid_sizes.size == 0:
            return 0.0, 0.0, False, False

        off_inputs = self._extractor.get_offensive_inputs()
        def_inputs = self._extractor.get_defensive_inputs()
        gate_inputs = self._extractor.get_gate_inputs()

        raw_thrust_off = self._thrust_offensive_engine.infer(off_inputs)
        raw_thrust_def = self._thrust_defensive_engine.infer(def_inputs)
        raw_turn_off = self._turn_offensive_engine.infer(off_inputs)
        raw_turn_def = self._turn_defensive_engine.infer(def_inputs)

        w_threat = max(
            0.0, min(float(self._threat_gate_engine.infer(gate_inputs)), 1.0)
        )

        blended_thrust = (w_threat * raw_thrust_def) + (
            (1.0 - w_threat) * raw_thrust_off
        )
        blended_turn = (w_threat * raw_turn_def) + (
            (1.0 - w_threat) * raw_turn_off
        )

        if np.isnan(blended_thrust) or np.isinf(blended_thrust):
            blended_thrust = 0.0
        if np.isnan(blended_turn) or np.isinf(blended_turn):
            blended_turn = 0.0

        physical_thrust = denorm_thrust(blended_thrust)
        physical_turn_rate = denorm_turn_rate(blended_turn)

        physical_thrust, physical_turn_rate = self._apply_volatility_damping(
            physical_thrust,
            physical_turn_rate,
        )

        thrust = max(-480.0, min(float(physical_thrust), 480.0))
        turn_rate = max(-180.0, min(float(physical_turn_rate), 180.0))

        fire = bool(self._determine_tactical_fire())
        drop_mine = False

        self._update_volatility_metrics(thrust, turn_rate)

        return thrust, turn_rate, fire, drop_mine

    @property
    @override
    def name(self) -> str:
        return "AsianAsteroidDestroyersFuzzyController"

    @property
    def mean_volatility(self) -> float:
        if self._processed_steps == 0:
            return 0.0
        return self._total_accumulated_volatility / self._processed_steps

    def _apply_volatility_damping(
        self,
        target_thrust: float,
        target_turn: float,
    ) -> tuple[float, float]:
        if self._last_thrust is None or self._last_turn is None:
            return target_thrust, target_turn

        if self._last_thrust == 0.0 and self._last_turn == 0.0:
            return target_thrust, target_turn

        d_thrust = target_thrust - self._last_thrust
        d_turn = target_turn - self._last_turn

        volatility = self.mean_volatility
        if np.isnan(volatility) or np.isinf(volatility):
            volatility = 0.0

        scale = max(0.1, min(1.0 - (volatility * self._damping_factor), 1.0))

        return (
            self._last_thrust + (d_thrust * scale),
            self._last_turn + (d_turn * scale),
        )

    def _determine_tactical_fire(self) -> bool:
        soonest = self._sa.agent_ship.soonest_impact_n_asteroids(1)
        if not soonest:
            nearest = self._sa.agent_ship.nearest_n_asteroids_wrapped(1)
            if not nearest:
                return False
            threat = nearest[0]
        else:
            threat = soonest[0]

        ship = self._sa.agent_ship
        p_bearing = self._extractor.get_predictive_bearing(threat)
        aim_err = abs((p_bearing - ship.heading + 180.0) % 360.0 - 180.0)

        return aim_err <= 25.0

    def _update_volatility_metrics(
        self,
        thrust: float,
        turn_rate: float,
    ) -> None:
        if self._last_thrust is not None and self._last_turn is not None:
            delta_thrust = abs(thrust - self._last_thrust)
            delta_turn = abs(turn_rate - self._last_turn)

            if not np.isnan(delta_thrust) and not np.isnan(delta_turn):
                self._total_accumulated_volatility += delta_thrust + delta_turn
                self._processed_steps += 1

        self._last_thrust = thrust
        self._last_turn = turn_rate
