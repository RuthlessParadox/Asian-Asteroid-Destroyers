from __future__ import annotations

from typing import TYPE_CHECKING

from asian_asteroid_destroyers.scenarios import load_scenario_suite

if TYPE_CHECKING:
    from asian_asteroid_destroyers.scenarios import ScenarioConfig

_AsteroidStateDict = dict[str, tuple[float, float] | float | int]


def test_load_scenario_suite_deterministic() -> None:
    config: ScenarioConfig = {
        "is_deterministic": True,
        "seed": 42,
        "json_data": {
            "map_size": [800, 600],
            "ship_spawn": [400.0, 300.0],
            "ship_heading": 90.0,
            "scenarios": [
                {
                    "name": "Deterministic Test Scenario",
                    "time_limit": 60.0,
                    "stop_if_no_ammo": True,
                    "ammo_limit_multiplier": 1.5,
                    "asteroid_groups": [
                        {
                            "count": 2,
                            "size": 2,
                            "speed_range": [50.0, 150.0],
                            "angle_range": [0.0, 180.0],
                            "pos_zones": [
                                {
                                    "x_range": [100.0, 200.0],
                                    "y_range": [150.0, 250.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }

    scenarios = load_scenario_suite(config)
    assert len(scenarios) == 1
    scene = scenarios[0]

    assert scene.name == "Deterministic Test Scenario"
    assert scene.time_limit == 60.0
    assert scene.stop_if_no_ammo is True

    asteroid_states: list[  # pyright: ignore[reportUnknownVariableType]
        _AsteroidStateDict
    ] = scene.asteroid_states  # pyright: ignore[reportUnknownMemberType]
    assert len(asteroid_states) == 2

    for state in asteroid_states:
        assert state["speed"] == 100.0
        assert state["angle"] == 90.0
        assert state["position"] == (150.0, 200.0)
        assert state["size"] == 2


def test_load_scenario_suite_stochastic_seeded() -> None:
    """Test stochastic scenario generation with a fixed seed for repeatability."""
    config: ScenarioConfig = {
        "is_deterministic": False,
        "seed": 123,
        "json_data": {
            "map_size": [1000, 800],
            "ship_spawn": [500.0, 400.0],
            "ship_heading": 0.0,
            "scenarios": [
                {
                    "name": "Stochastic Seeded Scenario",
                    "time_limit": 120.0,
                    "stop_if_no_ammo": False,
                    "ammo_limit_multiplier": None,
                    "asteroid_groups": [
                        {
                            "count": 3,
                            "size": 1,
                            "speed_range": [20.0, 40.0],
                            "angle_range": [0.0, 360.0],
                            "pos_zones": [
                                {
                                    "x_range": [0.0, 500.0],
                                    "y_range": [0.0, 400.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }

    scenarios_run_1 = load_scenario_suite(config)
    scenarios_run_2 = load_scenario_suite(config)

    assert len(scenarios_run_1) == 1
    assert len(scenarios_run_2) == 1

    states_1: list[  # pyright: ignore[reportUnknownVariableType]
        _AsteroidStateDict
    ] = (  # pyright: ignore[reportUnknownMemberType]
        scenarios_run_1[0].asteroid_states
    )
    states_2: list[  # pyright: ignore[reportUnknownVariableType]
        _AsteroidStateDict
    ] = (  # pyright: ignore[reportUnknownMemberType]
        scenarios_run_2[0].asteroid_states
    )

    assert len(states_1) == 3
    assert len(states_2) == 3

    for state_1, state_2 in zip(states_1, states_2):
        assert state_1["position"] == state_2["position"]
        assert state_1["speed"] == state_2["speed"]
        assert state_1["angle"] == state_2["angle"]
        assert state_1["size"] == state_2["size"]

    ships = scenarios_run_1[0].ships()
    assert len(ships) == 1
    assert ships[0].bullets_remaining == -1


def test_load_scenario_suite_multiple_zones_cycling() -> None:
    config: ScenarioConfig = {
        "is_deterministic": True,
        "json_data": {
            "map_size": [1000, 1000],
            "ship_spawn": [0.0, 0.0],
            "ship_heading": 45.0,
            "scenarios": [
                {
                    "name": "Multi Zone Test",
                    "time_limit": 30.0,
                    "stop_if_no_ammo": False,
                    "ammo_limit_multiplier": 1.0,
                    "asteroid_groups": [
                        {
                            "count": 3,
                            "size": 3,
                            "speed_range": [10.0, 10.0],
                            "angle_range": [45.0, 45.0],
                            "pos_zones": [
                                {
                                    "x_range": [0.0, 100.0],
                                    "y_range": [0.0, 100.0],
                                },
                                {
                                    "x_range": [200.0, 300.0],
                                    "y_range": [200.0, 300.0],
                                },
                            ],
                        }
                    ],
                }
            ],
        },
    }

    scenarios = load_scenario_suite(config)
    states: list[  # pyright: ignore[reportUnknownVariableType]
        _AsteroidStateDict
    ] = (  # pyright: ignore[reportUnknownMemberType]
        scenarios[0].asteroid_states
    )

    assert states[0]["position"] == (50.0, 50.0)
    assert states[1]["position"] == (250.0, 250.0)
    assert states[2]["position"] == (50.0, 50.0)


def test_load_scenario_suite_edge_cases_zero_time_and_null_ammo() -> None:
    config: ScenarioConfig = {
        "is_deterministic": True,
        "json_data": {
            "map_size": [500, 500],
            "ship_spawn": [250.0, 250.0],
            "ship_heading": 180.0,
            "scenarios": [
                {
                    "name": "Edge Cases Test",
                    "time_limit": 0.0,
                    "stop_if_no_ammo": False,
                    "ammo_limit_multiplier": None,
                    "asteroid_groups": [
                        {
                            "count": 1,
                            "size": 1,
                            "speed_range": [10.0, 10.0],
                            "angle_range": [0.0, 0.0],
                            "pos_zones": [
                                {
                                    "x_range": [0.0, 100.0],
                                    "y_range": [0.0, 100.0],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }

    scenarios = load_scenario_suite(config)
    assert len(scenarios) == 1
    scene = scenarios[0]

    assert scene.time_limit == 0.0

    ships = scene.ships()
    assert len(ships) == 1
    assert ships[0].bullets_remaining == -1
