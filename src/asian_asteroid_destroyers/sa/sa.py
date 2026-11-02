from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Generic, NamedTuple, TypeVar, cast

import numpy as np
import numpy.typing as npt

from asian_asteroid_destroyers.geometry import trim_angle

if TYPE_CHECKING:
    from collections.abc import Iterator

    from kesslergame.state_models import GameState, ShipState, ShipView


class SituationalAwareness:
    time: float
    agent_ship: AgentShip
    metrics: SituationalAwarenessMetrics
    size_ratio_threshold: Final[float]
    red_ships: list[OpponentShipView]
    map_diagonal_length: float

    _asteroid_count: int
    _bullet_count: int
    _mine_count: int

    _map_size: npt.NDArray[np.float64]

    _asteroid_positions: npt.NDArray[np.float64]
    _asteroid_velocities: npt.NDArray[np.float64]
    _asteroid_sizes: npt.NDArray[np.int8]
    _asteroid_radii: npt.NDArray[np.float64]

    _bullet_positions: npt.NDArray[np.float64]
    _bullet_velocities: npt.NDArray[np.float64]
    _bullet_headings: npt.NDArray[np.float64]

    _mine_positions: npt.NDArray[np.float64]
    _mine_fuse_times: npt.NDArray[np.float64]
    _mine_remaining_times: npt.NDArray[np.float64]

    def __init__(self) -> None:
        self.time = 0.0
        self.agent_ship = AgentShip()
        self.metrics = SituationalAwarenessMetrics()
        self.size_ratio_threshold = 2.0
        self.red_ships = []
        self.map_diagonal_length = 0.0

        self._asteroid_count = 0
        self._bullet_count = 0
        self._mine_count = 0

        self._map_size = np.zeros(2, dtype=np.float64)

        initial_capacity = 128

        self._asteroid_positions = np.zeros(
            (initial_capacity, 2),
            dtype=np.float64,
        )
        self._asteroid_velocities = np.zeros(
            (initial_capacity, 2),
            dtype=np.float64,
        )
        self._asteroid_sizes = np.zeros(initial_capacity, dtype=np.int8)
        self._asteroid_radii = np.zeros(initial_capacity, dtype=np.float64)

        self._bullet_positions = np.zeros(
            (initial_capacity, 2),
            dtype=np.float64,
        )
        self._bullet_velocities = np.zeros(
            (initial_capacity, 2),
            dtype=np.float64,
        )
        self._bullet_headings = np.zeros(initial_capacity, dtype=np.float64)

        self._mine_positions = np.zeros(
            (initial_capacity, 2),
            dtype=np.float64,
        )
        self._mine_fuse_times = np.zeros(initial_capacity, dtype=np.float64)
        self._mine_remaining_times = np.zeros(
            initial_capacity,
            dtype=np.float64,
        )

    def _ensure_asteroid_capacity(self, required_capacity: int) -> None:
        current_capacity = self._asteroid_sizes.size
        if required_capacity > current_capacity:
            new_capacity = max(current_capacity * 2, required_capacity)
            self._asteroid_positions = np.resize(
                self._asteroid_positions,
                (new_capacity, 2),
            )
            self._asteroid_velocities = np.resize(
                self._asteroid_velocities,
                (new_capacity, 2),
            )
            self._asteroid_sizes = np.resize(
                self._asteroid_sizes,
                new_capacity,
            )
            self._asteroid_radii = np.resize(
                self._asteroid_radii,
                new_capacity,
            )

    def _ensure_bullet_capacity(self, required_capacity: int) -> None:
        current_capacity = self._bullet_headings.size
        if required_capacity > current_capacity:
            new_capacity = max(current_capacity * 2, required_capacity)
            self._bullet_positions = np.resize(
                self._bullet_positions,
                (new_capacity, 2),
            )
            self._bullet_velocities = np.resize(
                self._bullet_velocities,
                (new_capacity, 2),
            )
            self._bullet_headings = np.resize(
                self._bullet_headings,
                new_capacity,
            )

    def _ensure_mine_capacity(self, required_capacity: int) -> None:
        current_capacity = self._mine_fuse_times.size
        if required_capacity > current_capacity:
            new_capacity = max(current_capacity * 2, required_capacity)
            self._mine_positions = np.resize(
                self._mine_positions,
                (new_capacity, 2),
            )
            self._mine_fuse_times = np.resize(
                self._mine_fuse_times,
                new_capacity,
            )
            self._mine_remaining_times = np.resize(
                self._mine_remaining_times,
                new_capacity,
            )

    def update(self, ship_state: ShipState, game_state: GameState) -> None:
        self.time = game_state.time

        if self.map_diagonal_length == 0.0:
            map_x, map_y = game_state.map_size
            self._map_size = np.array([map_x, map_y], dtype=np.float64)
            self.map_diagonal_length = math.hypot(map_x, map_y)

        asteroids_view = game_state.asteroids
        self._asteroid_count = len(asteroids_view)
        self._ensure_asteroid_capacity(self._asteroid_count)
        for index, asteroid in enumerate(asteroids_view):
            self._asteroid_positions[index, 0] = asteroid.position[0]
            self._asteroid_positions[index, 1] = asteroid.position[1]
            self._asteroid_velocities[index, 0] = asteroid.velocity[0]
            self._asteroid_velocities[index, 1] = asteroid.velocity[1]
            self._asteroid_sizes[index] = asteroid.size
            self._asteroid_radii[index] = asteroid.radius

        bullets_view = game_state.bullets
        self._bullet_count = len(bullets_view)
        self._ensure_bullet_capacity(self._bullet_count)
        for index, bullet in enumerate(bullets_view):
            self._bullet_positions[index, 0] = bullet.position[0]
            self._bullet_positions[index, 1] = bullet.position[1]
            self._bullet_velocities[index, 0] = bullet.velocity[0]
            self._bullet_velocities[index, 1] = bullet.velocity[1]
            self._bullet_headings[index] = bullet.heading

        mines_view = game_state.mines
        self._mine_count = len(mines_view)
        self._ensure_mine_capacity(self._mine_count)
        for index, mine in enumerate(mines_view):
            self._mine_positions[index, 0] = mine.position[0]
            self._mine_positions[index, 1] = mine.position[1]
            self._mine_fuse_times[index] = mine.fuse_time
            self._mine_remaining_times[index] = mine.remaining_time

        self.agent_ship.update(game_state, ship_state, self)

        self.red_ships.clear()
        self.red_ships = [
            OpponentShipView.from_view(ship)
            for ship in game_state.ships
            if ship.id != ship_state.id
        ]

        self.metrics.update(self)

    @property
    def map_size(self) -> npt.NDArray[np.float64]:
        return self._map_size

    @property
    def asteroid_positions(self) -> npt.NDArray[np.float64]:
        return self._asteroid_positions[: self._asteroid_count]

    @property
    def asteroid_velocities(self) -> npt.NDArray[np.float64]:
        return self._asteroid_velocities[: self._asteroid_count]

    @property
    def asteroid_sizes(self) -> npt.NDArray[np.int8]:
        return self._asteroid_sizes[: self._asteroid_count]

    @property
    def asteroid_radii(self) -> npt.NDArray[np.float64]:
        return self._asteroid_radii[: self._asteroid_count]

    @property
    def bullet_positions(self) -> npt.NDArray[np.float64]:
        return self._bullet_positions[: self._bullet_count]

    @property
    def bullet_velocities(self) -> npt.NDArray[np.float64]:
        return self._bullet_velocities[: self._bullet_count]

    @property
    def bullet_headings(self) -> npt.NDArray[np.float64]:
        return self._bullet_headings[: self._bullet_count]

    @property
    def mine_positions(self) -> npt.NDArray[np.float64]:
        return self._mine_positions[: self._mine_count]

    @property
    def mine_fuse_times(self) -> npt.NDArray[np.float64]:
        return self._mine_fuse_times[: self._mine_count]

    @property
    def mine_remaining_times(self) -> npt.NDArray[np.float64]:
        return self._mine_remaining_times[: self._mine_count]


class SituationalAwarenessMetrics:
    average_asteroid_size: float
    std_asteroid_size: float
    average_asteroid_speed: float
    std_asteroid_speed: float

    def __init__(self) -> None:
        self.average_asteroid_size = 0.0
        self.std_asteroid_size = 0.0
        self.average_asteroid_speed = 0.0
        self.std_asteroid_speed = 0.0

    def update(self, sa: SituationalAwareness) -> None:
        sizes = sa.asteroid_sizes
        if sizes.size > 0:
            self.average_asteroid_size = float(np.mean(sizes))
            self.std_asteroid_size = float(np.std(sizes))
            speeds = cast(
                npt.NDArray[np.float64],
                np.linalg.norm(sa.asteroid_velocities, axis=1),
            )
            self.average_asteroid_speed = float(np.mean(speeds))
            self.std_asteroid_speed = float(np.std(speeds))


class AsteroidView:
    __slots__: tuple[str, ...] = ("_sa", "_ship", "_index")
    _sa: Final[SituationalAwareness]
    _ship: Final[AgentShip]
    _index: Final[int]

    def __init__(
        self, sa: SituationalAwareness, ship: AgentShip, index: int
    ) -> None:
        self._sa = sa
        self._ship = ship
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def position(self) -> tuple[float, float]:
        return (
            cast(float, self._sa.asteroid_positions[self._index, 0]),
            cast(float, self._sa.asteroid_positions[self._index, 1]),
        )

    @property
    def velocity(self) -> tuple[float, float]:
        return (
            cast(float, self._sa.asteroid_velocities[self._index, 0]),
            cast(float, self._sa.asteroid_velocities[self._index, 1]),
        )

    @property
    def size(self) -> int:
        return cast(int, self._sa.asteroid_sizes[self._index])

    @property
    def radius(self) -> float:
        return cast(float, self._sa.asteroid_radii[self._index])

    @property
    def distance(self) -> float:
        return cast(float, self._ship.distances[self._index])

    @property
    def bearing(self) -> float:
        return cast(float, self._ship.bearings[self._index])

    @property
    def position_wrap(self) -> tuple[float, float]:
        wrap_matrix = self._ship.positions_wrap
        return (
            cast(float, wrap_matrix[self._index, 0]),
            cast(float, wrap_matrix[self._index, 1]),
        )

    @property
    def distance_wrap(self) -> float:
        return cast(float, self._ship.distances_wrap[self._index])

    @property
    def bearing_wrap(self) -> float:
        return cast(float, self._ship.bearings_wrap[self._index])

    @property
    def speed(self) -> float:
        return math.hypot(
            cast(float, self._sa.asteroid_velocities[self._index, 0]),
            cast(float, self._sa.asteroid_velocities[self._index, 1]),
        )

    @property
    def heading(self) -> float:
        return math.degrees(
            math.atan2(
                cast(float, self._sa.asteroid_velocities[self._index, 1]),
                cast(float, self._sa.asteroid_velocities[self._index, 0]),
            )
        )

    @property
    def tti(self) -> float | None:
        value = cast(float, self._ship.ttis[self._index])
        return None if math.isnan(value) else value


class BulletView:
    __slots__: tuple[str, ...] = ("_sa", "_ship", "_index")
    _sa: Final[SituationalAwareness]
    _ship: Final[AgentShip]
    _index: Final[int]

    def __init__(
        self, sa: SituationalAwareness, ship: AgentShip, index: int
    ) -> None:
        self._sa = sa
        self._ship = ship
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def position(self) -> tuple[float, float]:
        return (
            cast(float, self._sa.bullet_positions[self._index, 0]),
            cast(float, self._sa.bullet_positions[self._index, 1]),
        )

    @property
    def velocity(self) -> tuple[float, float]:
        return (
            cast(float, self._sa.bullet_velocities[self._index, 0]),
            cast(float, self._sa.bullet_velocities[self._index, 1]),
        )

    @property
    def heading(self) -> float:
        return cast(float, self._sa.bullet_headings[self._index])

    @property
    def distance(self) -> float:
        return cast(float, self._ship.bullet_distances[self._index])

    @property
    def bearing(self) -> float:
        return cast(float, self._ship.bullet_bearings[self._index])


class MineView:
    __slots__: tuple[str, ...] = ("_sa", "_ship", "_index")
    _sa: Final[SituationalAwareness]
    _ship: Final[AgentShip]
    _index: Final[int]

    def __init__(
        self, sa: SituationalAwareness, ship: AgentShip, index: int
    ) -> None:
        self._sa = sa
        self._ship = ship
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def position(self) -> tuple[float, float]:
        return (
            cast(float, self._sa.mine_positions[self._index, 0]),
            cast(float, self._sa.mine_positions[self._index, 1]),
        )

    @property
    def fuse_time(self) -> float:
        return cast(float, self._sa.mine_fuse_times[self._index])

    @property
    def remaining_time(self) -> float:
        return cast(float, self._sa.mine_remaining_times[self._index])

    @property
    def distance(self) -> float:
        return cast(float, self._ship.mine_distances[self._index])

    @property
    def bearing(self) -> float:
        return cast(float, self._ship.mine_bearings[self._index])


class OpponentShipView(NamedTuple):
    position: tuple[float, float]
    velocity: tuple[float, float]
    radius: float
    lives: int
    speed: float
    heading: float
    id: int

    @classmethod
    def from_view(cls, ship_view: ShipView) -> OpponentShipView:
        return cls(
            position=ship_view.position,
            velocity=ship_view.velocity,
            radius=ship_view.radius,
            lives=ship_view.lives_remaining,
            speed=ship_view.speed,
            heading=ship_view.heading,
            id=ship_view.id,
        )


class AgentShip:
    _time: float
    position: tuple[float, float]
    velocity: tuple[float, float]
    radius: float
    lives: int
    speed: float
    heading: float
    turn_rate: float
    acceleration: float
    bullets_remaining: int
    bullet_asteroid_equivalent_ratio: float
    asteroids: VirtualViewSequence[AsteroidView]
    bullets: VirtualViewSequence[BulletView]
    mines: VirtualViewSequence[MineView]
    _sa: SituationalAwareness | None
    _last_time: float
    last_position: tuple[float, float]
    last_heading: float
    last_speed: float
    _cache_distance: npt.NDArray[np.float64] | None
    _cache_bearing: npt.NDArray[np.float64] | None
    _cache_position_wrap: npt.NDArray[np.float64] | None
    _cache_distance_wrap: npt.NDArray[np.float64] | None
    _cache_bearing_wrap: npt.NDArray[np.float64] | None
    _cache_tti: npt.NDArray[np.float64] | None
    _cache_bullet_distance: npt.NDArray[np.float64] | None
    _cache_bullet_bearing: npt.NDArray[np.float64] | None
    _cache_mine_distance: npt.NDArray[np.float64] | None
    _cache_mine_bearing: npt.NDArray[np.float64] | None

    def __init__(self) -> None:
        self._time = -1.0
        self.position = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.radius = 0.0
        self.lives = 0
        self.speed = 0.0
        self.heading = 0.0
        self.turn_rate = 0.0
        self.acceleration = 0.0

        self.bullets_remaining = 0
        self.bullet_asteroid_equivalent_ratio = 0.0

        self.asteroids = VirtualViewSequence(AsteroidView)
        self.bullets = VirtualViewSequence(BulletView)
        self.mines = VirtualViewSequence(MineView)
        self._sa = None

        self._last_time = -1.0
        self.last_position = (0.0, 0.0)
        self.last_heading = 0.0
        self.last_speed = 0.0

        self._cache_distance = None
        self._cache_bearing = None
        self._cache_position_wrap = None
        self._cache_distance_wrap = None
        self._cache_bearing_wrap = None
        self._cache_tti = None

        self._cache_bullet_distance = None
        self._cache_bullet_bearing = None
        self._cache_mine_distance = None
        self._cache_mine_bearing = None

    def update(
        self,
        game_state: GameState,
        ship_state: ShipState,
        sa: SituationalAwareness,
    ) -> None:
        self._sa = sa
        self._last_time = self._time
        self.last_position = self.position
        self.last_heading = self.heading
        self.last_speed = self.speed

        self._cache_distance = None
        self._cache_bearing = None
        self._cache_position_wrap = None
        self._cache_distance_wrap = None
        self._cache_bearing_wrap = None
        self._cache_tti = None
        self._cache_bullet_distance = None
        self._cache_bullet_bearing = None
        self._cache_mine_distance = None
        self._cache_mine_bearing = None

        self._time = game_state.time
        self.position = ship_state.position
        self.velocity = ship_state.velocity
        self.radius = ship_state.radius
        self.lives = ship_state.lives_remaining
        self.speed = ship_state.speed
        self.heading = ship_state.heading

        self.bullets_remaining = ship_state.bullets_remaining
        self.turn_rate = self._infer_turn_rate()
        self.acceleration = self._infer_acceleration()
        self.bullet_asteroid_equivalent_ratio = self._update_bullet_ratio()

        self.asteroids.bind(sa, self, sa.asteroid_sizes.size)
        self.bullets.bind(sa, self, sa.bullet_headings.size)
        self.mines.bind(sa, self, sa.mine_fuse_times.size)

    @property
    def distances(self) -> npt.NDArray[np.float64]:
        return self._get_distances()

    @property
    def bearings(self) -> npt.NDArray[np.float64]:
        return self._get_bearings()

    @property
    def positions_wrap(self) -> npt.NDArray[np.float64]:
        return self._get_positions_wrap()

    @property
    def distances_wrap(self) -> npt.NDArray[np.float64]:
        return self._get_distances_wrap()

    @property
    def bearings_wrap(self) -> npt.NDArray[np.float64]:
        return self._get_bearings_wrap()

    @property
    def ttis(self) -> npt.NDArray[np.float64]:
        return self._get_ttis()

    @property
    def bullet_distances(self) -> npt.NDArray[np.float64]:
        return self._get_bullet_distances()

    @property
    def bullet_bearings(self) -> npt.NDArray[np.float64]:
        return self._get_bullet_bearings()

    @property
    def mine_distances(self) -> npt.NDArray[np.float64]:
        return self._get_mine_distances()

    @property
    def mine_bearings(self) -> npt.NDArray[np.float64]:
        return self._get_mine_bearings()

    def _infer_turn_rate(self) -> float:
        dt = self._time - self._last_time
        return (
            trim_angle(self.heading - self.last_heading) / dt
            if dt > 0
            else 0.0
        )

    def _infer_acceleration(self) -> float:
        dt = self._time - self._last_time
        return (self.speed - self.last_speed) / dt if dt > 0 else 0.0

    def _get_distances(self) -> npt.NDArray[np.float64]:
        if self._cache_distance is not None:
            return self._cache_distance

        assert self._sa is not None
        computed = cast(
            npt.NDArray[np.float64],
            np.linalg.norm(
                self._sa.asteroid_positions - np.array(self.position),
                axis=1,
            ),
        )
        self._cache_distance = computed
        return computed

    def _get_bearings(self) -> npt.NDArray[np.float64]:
        if self._cache_bearing is not None:
            return self._cache_bearing

        assert self._sa is not None
        delta = self._sa.asteroid_positions - np.array(self.position)
        computed = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        self._cache_bearing = computed
        return computed

    def _get_positions_wrap(self) -> npt.NDArray[np.float64]:
        if self._cache_position_wrap is not None:
            return self._cache_position_wrap

        assert self._sa is not None
        map_size = self._sa.map_size
        delta = self._sa.asteroid_positions - np.array(self.position)
        computed = np.where(
            np.abs(delta) > map_size / 2.0,
            self._sa.asteroid_positions - np.sign(delta) * map_size,
            self._sa.asteroid_positions,
        )
        self._cache_position_wrap = computed
        return computed

    def _get_distances_wrap(self) -> npt.NDArray[np.float64]:
        if self._cache_distance_wrap is not None:
            return self._cache_distance_wrap

        computed = cast(
            npt.NDArray[np.float64],
            np.linalg.norm(
                self._get_positions_wrap() - np.array(self.position),
                axis=1,
            ),
        )
        self._cache_distance_wrap = computed
        return computed

    def _get_bearings_wrap(self) -> npt.NDArray[np.float64]:
        if self._cache_bearing_wrap is not None:
            return self._cache_bearing_wrap

        delta = self._get_positions_wrap() - np.array(self.position)
        computed = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        self._cache_bearing_wrap = computed
        return computed

    def _get_ttis(self) -> npt.NDArray[np.float64]:
        if self._cache_tti is not None:
            return self._cache_tti

        assert self._sa is not None
        asteroid_count = self._sa.asteroid_sizes.size
        ttis = np.full(asteroid_count, np.nan, dtype=np.float64)
        if asteroid_count == 0:
            self._cache_tti = ttis
            return ttis

        position_asteroids = self._get_positions_wrap()
        position_asteroids_x = position_asteroids[:, 0]
        position_asteroids_y = position_asteroids[:, 1]
        velocity_asteroids = self._sa.asteroid_velocities
        position_self = np.array(self.position)
        velocity_self = np.array(self.velocity)
        radii_sums = self._sa.asteroid_radii + self.radius

        relative_velocities = velocity_self - velocity_asteroids
        zero_mask = np.equal(relative_velocities, 0.0)
        relative_velocities = np.where(zero_mask, 1e-10, relative_velocities)
        relative_velocities_x = relative_velocities[:, 0]
        relative_velocities_y = relative_velocities[:, 1]

        x_delta = np.subtract(
            cast(float, position_self[0]),
            position_asteroids_x,
        )
        intersection_times_x_1 = np.divide(
            -np.add(x_delta, radii_sums),
            relative_velocities_x,
        )
        intersection_times_x_2 = np.divide(
            -np.subtract(x_delta, radii_sums),
            relative_velocities_x,
        )
        intersection_time_x_highs = np.maximum(
            intersection_times_x_1,
            intersection_times_x_2,
        )
        intersection_time_x_lows = np.minimum(
            intersection_times_x_1,
            intersection_times_x_2,
        )

        y_delta = np.subtract(
            cast(float, position_self[1]),
            position_asteroids_y,
        )
        intersection_times_y_1 = np.divide(
            -np.add(y_delta, radii_sums),
            relative_velocities_y,
        )
        intersection_times_y_2 = np.divide(
            -np.subtract(y_delta, radii_sums),
            relative_velocities_y,
        )
        intersection_time_y_highs = np.maximum(
            intersection_times_y_1,
            intersection_times_y_2,
        )
        intersection_time_y_lows = np.minimum(
            intersection_times_y_1,
            intersection_times_y_2,
        )

        intersections = (
            intersection_time_x_highs > intersection_time_y_lows
        ) & (intersection_time_y_highs > intersection_time_x_lows)
        computed_ttis = np.minimum(
            np.maximum(intersection_time_x_highs, intersection_time_y_highs),
            np.maximum(intersection_time_x_lows, intersection_time_y_lows),
        )

        ttis = np.where(
            intersections & (computed_ttis >= 0.0),
            computed_ttis,
            np.nan,
        )
        self._cache_tti = ttis
        return ttis

    def _get_bullet_distances(self) -> npt.NDArray[np.float64]:
        if self._cache_bullet_distance is not None:
            return self._cache_bullet_distance

        assert self._sa is not None
        if self._sa.bullet_headings.size == 0:
            return np.empty(0, dtype=np.float64)

        computed = cast(
            npt.NDArray[np.float64],
            np.linalg.norm(
                self._sa.bullet_positions - np.array(self.position),
                axis=1,
            ),
        )
        self._cache_bullet_distance = computed
        return computed

    def _get_bullet_bearings(self) -> npt.NDArray[np.float64]:
        if self._cache_bullet_bearing is not None:
            return self._cache_bullet_bearing

        assert self._sa is not None
        if self._sa.bullet_headings.size == 0:
            return np.empty(0, dtype=np.float64)

        delta = self._sa.bullet_positions - np.array(self.position)
        computed = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        self._cache_bullet_bearing = computed
        return computed

    def _get_mine_distances(self) -> npt.NDArray[np.float64]:
        if self._cache_mine_distance is not None:
            return self._cache_mine_distance

        assert self._sa is not None
        if self._sa.mine_fuse_times.size == 0:
            return np.empty(0, dtype=np.float64)

        computed = cast(
            npt.NDArray[np.float64],
            np.linalg.norm(
                self._sa.mine_positions - np.array(self.position),
                axis=1,
            ),
        )
        self._cache_mine_distance = computed
        return computed

    def _get_mine_bearings(self) -> npt.NDArray[np.float64]:
        if self._cache_mine_bearing is not None:
            return self._cache_mine_bearing

        assert self._sa is not None
        if self._sa.mine_fuse_times.size == 0:
            return np.empty(0, dtype=np.float64)

        delta = self._sa.mine_positions - np.array(self.position)
        computed = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        self._cache_mine_bearing = computed
        return computed

    def within_radius(self, radius: float) -> list[AsteroidView]:
        asteroid_distances = self.distances
        return [
            asteroid
            for asteroid in self.asteroids
            if asteroid_distances[asteroid.index] < radius
        ]

    def within_radius_wrap(self, radius: float) -> list[AsteroidView]:
        asteroid_distances_wrapped = self.distances_wrap
        return [
            asteroid
            for asteroid in self.asteroids
            if asteroid_distances_wrapped[asteroid.index] < radius
        ]

    def nearest_n_asteroids(self, n_asteroids: int) -> list[AsteroidView]:
        indices = cast(
            list[int],
            np.argsort(self.distances)[:n_asteroids].tolist(),
        )
        return [self.asteroids[index] for index in indices]

    def nearest_n_asteroids_wrapped(
        self,
        n_asteroids: int,
    ) -> list[AsteroidView]:
        indices = cast(
            list[int],
            np.argsort(self.distances_wrap)[:n_asteroids].tolist(),
        )
        return [self.asteroids[index] for index in indices]

    def soonest_impact_n_asteroids(
        self,
        n_asteroids: int,
    ) -> list[AsteroidView]:
        ttis = self.ttis
        valid_mask = ~np.isnan(ttis)
        if not np.any(valid_mask):
            return []

        valid_indices = np.where(valid_mask)[0]
        sorted_indices = valid_indices[np.argsort(ttis[valid_mask])][
            :n_asteroids
        ]

        indices = cast(list[int], sorted_indices.tolist())
        return [self.asteroids[index] for index in indices]

    def _update_bullet_ratio(self) -> float:
        assert self._sa is not None
        if self._sa.asteroid_sizes.size == 0:
            return 0.0

        sizes = self._sa.asteroid_sizes
        shots_needed = (
            cast(int, np.sum(np.equal(sizes, 1))) * 1
            + cast(int, np.sum(np.equal(sizes, 2))) * 4
            + cast(int, np.sum(np.equal(sizes, 3))) * 13
            + cast(int, np.sum(np.equal(sizes, 4))) * 40
        )
        return (
            (self.bullets_remaining - shots_needed) / shots_needed
            if shots_needed > 0
            else 0.0
        )


T = TypeVar("T", AsteroidView, BulletView, MineView)


class VirtualViewSequence(Generic[T]):
    __slots__: tuple[str, ...] = ("_view_class", "_sa", "_ship", "_size")
    _view_class: Final[type[T]]
    _sa: SituationalAwareness | None
    _ship: AgentShip | None
    _size: int

    def __init__(self, view_class: type[T]) -> None:
        self._view_class = view_class
        self._sa = None
        self._ship = None
        self._size = 0

    def bind(
        self,
        sa: SituationalAwareness,
        ship: AgentShip,
        size: int,
    ) -> None:
        self._sa = sa
        self._ship = ship
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> T:
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError(
                "Virtual sequence index out of tactical range bounds."
            )

        assert self._sa is not None
        assert self._ship is not None
        return self._view_class(self._sa, self._ship, index)

    def __iter__(self) -> Iterator[T]:
        for i in range(self._size):
            yield self[i]
