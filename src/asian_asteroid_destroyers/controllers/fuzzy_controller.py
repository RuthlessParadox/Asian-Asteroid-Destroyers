from __future__ import annotations

from typing import TYPE_CHECKING, Final

from kesslergame import KesslerController
from typing_extensions import override

from asian_asteroid_destroyers.chromosomes import compile_subsystem
from asian_asteroid_destroyers.sa import SituationalAwareness

if TYPE_CHECKING:
    from kesslergame.state_models import GameState, ShipState

    from asian_asteroid_destroyers.chromosomes import ControllerChromosome
    from asian_asteroid_destroyers.fuzzy import TSKInferenceEngine


class _TacticalFeatureExtractor:
    __slots__: tuple[str, ...] = ("_sa",)
    _sa: Final[SituationalAwareness]

    def __init__(self, sa: SituationalAwareness) -> None:
        self._sa = sa

    def get_thrust_inputs(self) -> list[float]:
        """Constructs input vector for the thrust inference engine.

        Matrix Layout:: Soonest impact asteroid distance (wrapped): Soonest impact asteroid time-to-impact (TTI): Current ship linear speed
        """
        soonest = self._sa.agent_ship.soonest_impact_n_asteroids(1)

        if not soonest:
            return [
                self._sa.map_diagonal_length,
                10.0,
                self._sa.agent_ship.speed,
            ]

        target_asteroid = soonest[0]
        tti = target_asteroid.tti
        resolved_tti = tti if tti is not None else 10.0

        return [
            target_asteroid.distance_wrap,
            resolved_tti,
            self._sa.agent_ship.speed,
        ]

    def get_turn_inputs(self) -> list[float]:
        """Constructs input vector for the turn rate inference engine.

        Matrix Layout:: Bearing angle to target threat (degrees wrapped): Soonest impact asteroid time-to-impact (TTI): Current ship angular turn rate
        """
        soonest = self._sa.agent_ship.soonest_impact_n_asteroids(1)

        if not soonest:
            nearest = self._sa.agent_ship.nearest_n_asteroids_wrapped(1)
            if not nearest:
                return [0.0, 10.0, self._sa.agent_ship.turn_rate]
            target_asteroid = nearest[0]
            resolved_tti = 10.0
        else:
            target_asteroid = soonest[0]
            tti = target_asteroid.tti
            resolved_tti = tti if tti is not None else 10.0

        return [
            target_asteroid.bearing_wrap,
            resolved_tti,
            self._sa.agent_ship.turn_rate,
        ]

    def get_fire_inputs(self) -> list[float]:
        """Constructs input vector for the firing command inference engine.

        Matrix Layout:: Bearing angle to targeting threat (degrees wrapped): Distance to targeted threat: Tactical ammunition surplus ratio
        """
        nearest = self._sa.agent_ship.nearest_n_asteroids_wrapped(1)

        if not nearest:
            return [0.0, self._sa.map_diagonal_length, 0.0]

        target_asteroid = nearest[0]
        return [
            target_asteroid.bearing_wrap,
            target_asteroid.distance_wrap,
            self._sa.agent_ship.bullet_asteroid_equivalent_ratio,
        ]


class FuzzyController(KesslerController):
    __slots__: tuple[str, ...] = (
        "_sa",
        "_extractor",
        "_thrust_engine",
        "_turn_engine",
        "_fire_engine",
        "_emergency_tti_threshold",
        "_last_thrust",
        "_last_turn",
        "_total_accumulated_volatility",
        "_processed_steps",
    )

    _sa: Final[SituationalAwareness]
    _extractor: Final[_TacticalFeatureExtractor]

    _thrust_engine: Final[TSKInferenceEngine]
    _turn_engine: Final[TSKInferenceEngine]
    _fire_engine: Final[TSKInferenceEngine]

    _emergency_tti_threshold: Final[float]

    _last_thrust: float | None
    _last_turn: float | None
    _total_accumulated_volatility: float
    _processed_steps: int

    def __init__(self, chromosome: ControllerChromosome) -> None:
        super().__init__()

        self._sa = SituationalAwareness()
        self._extractor = _TacticalFeatureExtractor(self._sa)

        self._thrust_engine = compile_subsystem(chromosome.thrust_blueprint)
        self._turn_engine = compile_subsystem(chromosome.turn_blueprint)
        self._fire_engine = compile_subsystem(chromosome.fire_blueprint)

        self._emergency_tti_threshold = 0.45

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

        thrust_inputs = self._extractor.get_thrust_inputs()
        turn_inputs = self._extractor.get_turn_inputs()
        fire_inputs = self._extractor.get_fire_inputs()

        thrust = self._thrust_engine.infer(thrust_inputs)
        turn_rate = self._turn_engine.infer(turn_inputs)
        fire_output = self._fire_engine.infer(fire_inputs)

        fire = fire_output > 0.0
        drop_mine = False

        # --- Emergency Fire-Override Fallback Layer ---
        # Force a fire directive if an asteroid is on an imminent crash course,
        # shielding the ship during early un-optimized genetic epochs.
        soonest_asteroids = self._sa.agent_ship.soonest_impact_n_asteroids(1)
        if soonest_asteroids:
            imminent_threat = soonest_asteroids[0]
            if imminent_threat.tti is not None:
                if imminent_threat.tti <= self._emergency_tti_threshold:
                    # Panic override: point-blank blast threat mitigation
                    fire = True

        if self._last_thrust is not None and self._last_turn is not None:
            delta_thrust = abs(thrust - self._last_thrust)
            delta_turn = abs(turn_rate - self._last_turn)
            self._total_accumulated_volatility += delta_thrust + delta_turn
            self._processed_steps += 1

        self._last_thrust = thrust
        self._last_turn = turn_rate

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
