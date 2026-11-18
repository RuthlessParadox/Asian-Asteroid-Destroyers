from __future__ import annotations

import math
import sys
from types import ModuleType

import pytest


def _mock_trim_angle(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


geometry_module = ModuleType("asian_asteroid_destroyers.geometry")
geometry_module.trim_angle = (  # pyright: ignore[reportAttributeAccessIssue]
    _mock_trim_angle
)
sys.modules["asian_asteroid_destroyers.geometry"] = geometry_module

from kesslergame.state_models import GameState, ShipState  # noqa: E402

from asian_asteroid_destroyers.sa import (  # noqa: E402
    OpponentShipView,
    SituationalAwareness,
    SituationalAwarenessMetrics,
)


def _create_ship_data(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    speed: float = 0.0,
    heading: float = 0.0,
    mass: float = 100.0,
    radius: float = 10.0,
    ship_id: int = 1,
    team: int = 1,
    is_respawning: bool = False,
    lives_remaining: int = 3,
    deaths: int = 0,
    bullets_remaining: int = 10,
    mines_remaining: int = 2,
    can_fire: bool = True,
    fire_cooldown: float = 0.0,
    fire_rate: float = 0.5,
    can_deploy_mine: bool = True,
    mine_cooldown: float = 0.0,
    mine_deploy_rate: float = 1.0,
    respawn_time_left: float = 0.0,
    respawn_time: float = 3.0,
    thrust_range: tuple[float, float] = (-10.0, 10.0),
    turn_rate_range: tuple[float, float] = (-180.0, 180.0),
    max_speed: float = 50.0,
    drag: float = 0.1,
) -> list[float | int | bool]:
    return [
        x,
        y,
        vx,
        vy,
        speed,
        heading,
        mass,
        radius,
        ship_id,
        team,
        is_respawning,
        lives_remaining,
        deaths,
        bullets_remaining,
        mines_remaining,
        can_fire,
        fire_cooldown,
        fire_rate,
        can_deploy_mine,
        mine_cooldown,
        mine_deploy_rate,
        respawn_time_left,
        respawn_time,
        thrust_range[0],
        thrust_range[1],
        turn_rate_range[0],
        turn_rate_range[1],
        max_speed,
        drag,
    ]


def _create_asteroid_data(
    x: float = 100.0,
    y: float = 100.0,
    vx: float = 1.0,
    vy: float = -1.0,
    size: int = 2,
    mass: float = 50.0,
    radius: float = 15.0,
) -> list[float | int]:
    return [x, y, vx, vy, size, mass, radius]


def _create_bullet_data(
    x: float = 50.0,
    y: float = 50.0,
    vx: float = 10.0,
    vy: float = 10.0,
    tail_dx: float = -2.0,
    tail_dy: float = -2.0,
    heading: float = 45.0,
    mass: float = 1.0,
    length: float = 5.0,
) -> list[float]:
    return [x, y, vx, vy, tail_dx, tail_dy, heading, mass, length]


def _create_mine_data(
    x: float = 200.0,
    y: float = 200.0,
    mass: float = 10.0,
    fuse_time: float = 5.0,
    remaining_time: float = 4.5,
) -> list[float]:
    return [x, y, mass, fuse_time, remaining_time]


@pytest.fixture
def sample_game_state() -> GameState:
    ship_data_1 = _create_ship_data(x=0.0, y=0.0, ship_id=1, team=1)
    ship_data_2 = _create_ship_data(x=500.0, y=500.0, ship_id=2, team=2)
    asteroid_data_1 = _create_asteroid_data(
        x=100.0,
        y=100.0,
        vx=10.0,
        vy=0.0,
        size=2,
        radius=15.0,
    )
    asteroid_data_2 = _create_asteroid_data(
        x=800.0,
        y=800.0,
        vx=0.0,
        vy=10.0,
        size=1,
        radius=10.0,
    )
    bullet_data_1 = _create_bullet_data(x=50.0, y=50.0, heading=90.0)
    mine_data_1 = _create_mine_data(x=150.0, y=150.0)

    return GameState(
        ships=[ship_data_1, ship_data_2],
        asteroids=[asteroid_data_1, asteroid_data_2],
        bullets=[bullet_data_1],
        mines=[mine_data_1],
        map_size=(1000, 1000),
        time_limit=60.0,
        time=1.0,
        frame=60,
        delta_time=1.0 / 60.0,
        frame_rate=60.0,
        random_asteroid_splits=False,
        competition_safe_mode=False,
    )


@pytest.fixture
def sample_ship_state(sample_game_state: GameState) -> ShipState:
    return ShipState(sample_game_state.compact["ships"][0])


def test_situational_awareness_init() -> None:
    sa = SituationalAwareness()
    assert sa.time == 0.0
    assert sa.size_ratio_threshold == 2.0
    assert sa.red_ships == []
    assert sa.map_diagonal_length == 0.0
    assert sa.map_width == 0.0
    assert sa.map_height == 0.0


def test_situational_awareness_update(
    sample_game_state: GameState,
    sample_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    sa.update(sample_ship_state, sample_game_state)

    assert sa.time == 1.0
    assert sa.map_width == 1000.0
    assert sa.map_height == 1000.0
    assert math.isclose(sa.map_diagonal_length, math.hypot(1000.0, 1000.0))
    assert len(sa.red_ships) == 1
    assert sa.red_ships[0].id == 2
    assert len(sa.asteroid_positions) == 2
    assert len(sa.bullet_positions) == 1
    assert len(sa.mine_positions) == 1


def test_capacity_resizing(
    sample_game_state: GameState, sample_ship_state: ShipState
) -> None:
    sa = SituationalAwareness()

    many_asteroids = [
        _create_asteroid_data(x=float(i), y=float(i)) for i in range(200)
    ]
    sample_game_state.add_asteroids(many_asteroids)

    sa.update(sample_ship_state, sample_game_state)

    assert sa.asteroid_positions.shape[0] == 202


def test_metrics_update(
    sample_game_state: GameState,
    sample_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    sa.update(sample_ship_state, sample_game_state)

    metrics = sa.metrics
    assert metrics.average_asteroid_size == 1.5
    assert metrics.std_asteroid_size > 0.0
    assert metrics.average_asteroid_speed > 0.0
    assert metrics.std_asteroid_speed >= 0.0

    empty_sa = SituationalAwareness()
    empty_metrics = SituationalAwarenessMetrics()
    empty_metrics.update(empty_sa)
    assert empty_metrics.average_asteroid_size == 0.0


def test_asteroid_view(
    sample_game_state: GameState,
    sample_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    sa.update(sample_ship_state, sample_game_state)

    asteroid = sa.agent_ship.asteroids[0]
    assert asteroid.index == 0
    assert asteroid.position == (100.0, 100.0)
    assert asteroid.velocity == (10.0, 0.0)
    assert asteroid.size == 2
    assert asteroid.radius == 15.0
    assert math.isclose(asteroid.distance, math.hypot(100.0, 100.0))
    assert math.isclose(
        asteroid.bearing,
        math.degrees(math.atan2(100.0, 100.0)),
    )
    assert asteroid.position_wrap == (100.0, 100.0)
    assert math.isclose(asteroid.distance_wrap, asteroid.distance)
    assert math.isclose(asteroid.bearing_wrap, asteroid.bearing)
    assert math.isclose(asteroid.speed, 10.0)
    assert math.isclose(asteroid.heading, 0.0)
    assert asteroid.tti is None


def test_bullet_and_mine_views(
    sample_game_state: GameState,
    sample_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    sa.update(sample_ship_state, sample_game_state)

    bullet = sa.agent_ship.bullets[0]
    assert bullet.index == 0
    assert bullet.position == (50.0, 50.0)
    assert bullet.velocity == (10.0, 10.0)
    assert math.isclose(bullet.heading, 90.0)
    assert bullet.distance > 0.0
    assert bullet.bearing != 0.0

    mine = sa.agent_ship.mines[0]
    assert mine.index == 0
    assert mine.position == (150.0, 150.0)
    assert mine.fuse_time == 5.0
    assert mine.remaining_time == 4.5
    assert mine.distance > 0.0
    assert mine.bearing != 0.0


def test_opponent_ship_view(sample_game_state: GameState) -> None:
    ship_view = sample_game_state.ships[1]
    opponent = OpponentShipView.from_view(ship_view)
    assert opponent.id == 2
    assert opponent.position == (500.0, 500.0)
    assert opponent.lives == 3


def test_agent_ship_inference_and_queries(
    sample_game_state: GameState,
    sample_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    sa.update(sample_ship_state, sample_game_state)

    ship = sa.agent_ship
    assert ship.turn_rate == 0.0
    assert ship.acceleration == 0.0

    sample_game_state.time = 2.0
    sa.update(sample_ship_state, sample_game_state)

    within_r = ship.within_radius(200.0)
    assert len(within_r) == 1

    within_r_wrap = ship.within_radius_wrapper(200.0)
    assert len(within_r_wrap) == 1

    nearest = ship.nearest_n_asteroids(1)
    assert len(nearest) == 1

    nearest_wrap = ship.nearest_n_asteroids_wrapped(1)
    assert len(nearest_wrap) == 1

    impact = ship.soonest_impact_n_asteroids(1)
    assert isinstance(impact, list)

    ratio = ship.bullet_asteroid_equivalent_ratio
    assert ratio > 0.0


def test_virtual_view_sequence_exceptions(
    sample_game_state: GameState,
    sample_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    sa.update(sample_ship_state, sample_game_state)

    asteroids = sa.agent_ship.asteroids
    assert len(asteroids) == 2

    last_item = asteroids[-1]
    assert last_item.index == 1

    with pytest.raises(IndexError):
        _ = asteroids[5]

    with pytest.raises(IndexError):
        _ = asteroids[-5]
