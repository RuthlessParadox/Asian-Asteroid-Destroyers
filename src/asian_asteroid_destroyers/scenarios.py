from __future__ import annotations

from typing import TYPE_CHECKING, Final

from kesslergame import Scenario

if TYPE_CHECKING:
    import numpy as np

__all__ = ["create_training_portfolio"]


def create_training_portfolio(rng: np.random.Generator) -> list[Scenario]:
    scenarios: list[Scenario] = []

    map_size: Final[tuple[int, int]] = (1000, 800)
    ship_spawn: Final[tuple[float, float]] = (500.0, 400.0)
    ship_heading: Final[float] = 90.0

    # -------------------------------------------------------------------------
    # Scenario 1: Accuracy & Target Tracking Drill
    # -------------------------------------------------------------------------
    scenario_1_asteroid_states: list[
        dict[str, tuple[float, float] | int | float]
    ] = []

    for _ in range(5):
        pos_x = (
            rng.uniform(100.0, 300.0)
            if rng.random() < 0.5
            else rng.uniform(700.0, 900.0)
        )
        pos_y = (
            rng.uniform(100.0, 200.0)
            if rng.random() < 0.5
            else rng.uniform(600.0, 700.0)
        )
        speed = rng.uniform(10.0, 30.0)
        angle = rng.uniform(0.0, 360.0)

        scenario_1_asteroid_states.append(
            {
                "position": (pos_x, pos_y),
                "speed": speed,
                "angle": angle,
                "size": 4,
            }
        )

    scenarios.append(
        Scenario(
            name="XFC_Drill_Tracking_Accuracy",
            asteroid_states=scenario_1_asteroid_states,
            ship_states=[{"position": ship_spawn, "angle": ship_heading}],
            map_size=map_size,
            time_limit=60.0,
            stop_if_no_ammo=True,
            ammo_limit_multiplier=2.0,
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 2: High-Velocity Evasion Minefield
    # -------------------------------------------------------------------------
    scenario_2_asteroid_states: list[
        dict[str, tuple[float, float] | int | float]
    ] = []

    for _ in range(12):
        pos_x = rng.uniform(50.0, 950.0)
        pos_y = (
            rng.uniform(50.0, 250.0)
            if rng.random() < 0.5
            else rng.uniform(550.0, 750.0)
        )
        speed = rng.uniform(80.0, 150.0)
        angle = rng.uniform(0.0, 360.0)

        scenario_2_asteroid_states.append(
            {
                "position": (pos_x, pos_y),
                "speed": speed,
                "angle": angle,
                "size": 2,
            }
        )

    scenarios.append(
        Scenario(
            name="XFC_Drill_High_Velocity_Evasion",
            asteroid_states=scenario_2_asteroid_states,
            ship_states=[{"position": ship_spawn, "angle": ship_heading}],
            map_size=map_size,
            time_limit=45.0,
            stop_if_no_ammo=False,
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 3: Wrap-Around Spatial Challenge
    # -------------------------------------------------------------------------
    scenario_3_asteroid_states: list[
        dict[str, tuple[float, float] | int | float]
    ] = []

    for _ in range(8):
        if rng.random() < 0.5:
            pos_x = rng.uniform(10.0, 50.0)
        else:
            pos_x = rng.uniform(950.0, 990.0)
        pos_y = rng.uniform(10.0, 790.0)

        speed = rng.uniform(60.0, 100.0)
        angle = rng.uniform(0.0, 360.0)

        scenario_3_asteroid_states.append(
            {
                "position": (pos_x, pos_y),
                "speed": speed,
                "angle": angle,
                "size": 3,
            }
        )

    scenarios.append(
        Scenario(
            name="XFC_Drill_Wrapped_Spatial_Awareness",
            asteroid_states=scenario_3_asteroid_states,
            ship_states=[{"position": ship_spawn, "angle": ship_heading}],
            map_size=map_size,
            time_limit=60.0,
            stop_if_no_ammo=True,
            ammo_limit_multiplier=2.5,
        )
    )

    return scenarios
