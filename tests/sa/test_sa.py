import math

import pytest
from kesslergame.state_models import (
    GameState,
    ShipState,
)

from asian_asteroid_destroyers.sa import (
    AgentShip,
    MineView as SAMineView,
    SituationalAwareness,
    VirtualViewSequence,
)


@pytest.fixture
def empty_game_state() -> GameState:
    return GameState(
        ships=[],
        asteroids=[],
        bullets=[],
        mines=[],
        map_size=(1000, 800),
        time_limit=180.0,
        time=0.0,
        frame=0,
        delta_time=1 / 30,
        frame_rate=30.0,
        random_asteroid_splits=False,
        competition_safe_mode=False,
    )


@pytest.fixture
def standard_ship_data() -> list[float | int | bool]:
    return [
        100.0,
        200.0,
        10.0,
        0.0,
        10.0,
        45.0,
        100.0,
        15.0,
        1,
        0,
        False,
        3,
        0,
        100,
        5,
        True,
        0.0,
        0.1,
        True,
        0.0,
        0.5,
        0.0,
        3.0,
        0.0,
        100.0,
        -180.0,
        180.0,
        50.0,
        0.05,
    ]


@pytest.fixture
def standard_ship_state(
    standard_ship_data: list[float | int | bool],
) -> ShipState:
    return ShipState(standard_ship_data)


def test_situational_awareness_initial_state() -> None:
    sa = SituationalAwareness()
    assert sa.time == 0.0
    assert sa.map_diagonal_length == 0.0
    assert sa.asteroid_positions.shape == (0, 2)
    assert sa.bullet_positions.shape == (0, 2)
    assert sa.mine_positions.shape == (0, 2)

    assert sa.metrics.average_asteroid_size == 0.0
    assert sa.metrics.average_asteroid_speed == 0.0


def test_situational_awareness_update_and_capacity_expansion(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()

    target_count = 130
    for i in range(target_count):
        empty_game_state.add_asteroid(
            [float(i), float(i * 2), 5.0, -5.0, 2, 50.0, 12.0],
        )
        empty_game_state.add_bullet(
            [float(i), 10.0, 100.0, 0.0, -2.0, 0.0, 0.0, 1.0, 4.0],
        )
        empty_game_state.add_mine([float(i), 20.0, 20.0, 3.0, 3.0])

    empty_game_state.time = 12.5
    sa.update(standard_ship_state, empty_game_state)

    assert sa.time == 12.5
    assert sa.map_diagonal_length == math.hypot(1000, 800)

    assert sa.asteroid_positions.shape == (target_count, 2)
    assert sa.bullet_positions.shape == (target_count, 2)
    assert sa.mine_positions.shape == (target_count, 2)

    assert sa.asteroid_positions[5, 0] == 5.0
    assert sa.asteroid_positions[5, 1] == 10.0


def test_metrics_calculation(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()

    empty_game_state.add_asteroid([0.0, 0.0, 3.0, 0.0, 1, 10.0, 5.0])
    empty_game_state.add_asteroid([0.0, 0.0, 0.0, 4.0, 3, 30.0, 15.0])

    sa.update(standard_ship_state, empty_game_state)

    assert sa.metrics.average_asteroid_size == 2.0
    assert sa.metrics.std_asteroid_size == 1.0
    assert sa.metrics.average_asteroid_speed == 3.5
    assert pytest.approx(sa.metrics.std_asteroid_speed) == 0.5


def test_agent_ship_inference(
    empty_game_state: GameState,
    standard_ship_data: list[float | int | bool],
) -> None:
    sa = SituationalAwareness()
    ship_state = ShipState(standard_ship_data)

    empty_game_state.time = 0.0
    sa.update(ship_state, empty_game_state)

    empty_game_state.time = 0.1

    standard_ship_data[5] = 47.0
    standard_ship_data[4] = 11.0

    sa.update(ship_state, empty_game_state)

    assert math.isclose(sa.agent_ship.turn_rate, 20.0, rel_tol=1e-5)
    assert math.isclose(sa.agent_ship.acceleration, 10.0, rel_tol=1e-5)


def test_opponent_filtering(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()

    empty_game_state.update_ships(
        [[200.0, 200.0, 0.0, 0.0, 0.0, 0.0, 50.0, 10.0, 2, 1, False, 3, 0]],
    )

    sa.update(standard_ship_state, empty_game_state)
    assert len(sa.red_ships) == 1
    assert sa.red_ships[0].id == 2


def test_asteroid_view_properties(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    empty_game_state.add_asteroid([103.0, 204.0, 10.0, 0.0, 2, 40.0, 8.0])

    sa.update(standard_ship_state, empty_game_state)

    view = sa.agent_ship.asteroids[0]
    assert view.index == 0
    assert view.position == (103.0, 204.0)
    assert view.velocity == (10.0, 0.0)
    assert view.size == 2
    assert view.radius == 8.0
    assert pytest.approx(view.distance) == 5.0
    assert pytest.approx(view.speed) == 10.0
    assert view.heading == 0.0


def test_bullet_and_mine_view_properties(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    empty_game_state.add_bullet(
        [100.0, 210.0, 0.0, 50.0, 0.0, -2.0, 90.0, 1.0, 5.0],
    )
    empty_game_state.add_mine([110.0, 200.0, 15.0, 4.0, 2.5])

    sa.update(standard_ship_state, empty_game_state)

    bullet_view = sa.agent_ship.bullets[0]
    assert bullet_view.position == (100.0, 210.0)
    assert pytest.approx(bullet_view.distance) == 10.0
    assert bullet_view.heading == 90.0

    mine_view = sa.agent_ship.mines[0]
    assert mine_view.position == (110.0, 200.0)
    assert pytest.approx(mine_view.distance) == 10.0
    assert mine_view.fuse_time == 4.0
    assert mine_view.remaining_time == 2.5


def test_spatial_radius_and_nearest_filters(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    empty_game_state.add_asteroid([103.0, 204.0, 0.0, 0.0, 1, 10.0, 2.0])
    empty_game_state.add_asteroid([100.0, 225.0, 0.0, 0.0, 1, 10.0, 2.0])

    sa.update(standard_ship_state, empty_game_state)

    close_asteroids = sa.agent_ship.within_radius(10.0)
    assert len(close_asteroids) == 1
    assert close_asteroids[0].index == 0

    nearest = sa.agent_ship.nearest_n_asteroids(2)
    assert len(nearest) == 2
    assert nearest[0].index == 0
    assert nearest[1].index == 1


def test_virtual_view_sequence_boundaries() -> None:
    sequence = VirtualViewSequence(SAMineView)
    sa = SituationalAwareness()
    ship = AgentShip()

    sequence.bind(sa, ship, 3)
    assert len(sequence) == 3

    item = sequence[-1]
    assert item.index == 2

    with pytest.raises(IndexError):
        _ = sequence[3]

    with pytest.raises(IndexError):
        _ = sequence[-4]


def test_time_to_impact_calculation(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()
    standard_ship_state.compact[0] = 500.0
    standard_ship_state.compact[1] = 400.0
    standard_ship_state.compact[2] = 0.0
    standard_ship_state.compact[3] = 0.0

    empty_game_state.add_asteroid([600.0, 400.0, -10.0, 0.0, 2, 20.0, 5.0])

    sa.update(standard_ship_state, empty_game_state)

    view = sa.agent_ship.asteroids[0]
    assert view.tti is not None
    assert pytest.approx(view.tti) == 8.0


def test_bullet_asteroid_equivalent_ratio(
    empty_game_state: GameState,
    standard_ship_state: ShipState,
) -> None:
    sa = SituationalAwareness()

    standard_ship_state.compact[13] = 60

    empty_game_state.add_asteroid([0.0, 0.0, 0.0, 0.0, 3, 10.0, 5.0])

    sa.update(standard_ship_state, empty_game_state)
    assert (
        pytest.approx(sa.agent_ship.bullet_asteroid_equivalent_ratio)
        == 47.0 / 13.0
    )
