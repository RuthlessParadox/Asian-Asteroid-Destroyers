from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from kesslergame import Scenario

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ScenarioConfig", "load_scenario_suite"]


class _ZoneDict(TypedDict):
    x_range: list[float]
    y_range: list[float]


class _GroupDict(TypedDict):
    count: int
    size: int
    speed_range: list[float]
    angle_range: list[float]
    pos_zones: list[_ZoneDict]


class _ScenarioJsonDict(TypedDict):
    name: str
    time_limit: float
    stop_if_no_ammo: bool
    ammo_limit_multiplier: float | None
    asteroid_groups: list[_GroupDict]


class _RootJsonSchema(TypedDict):
    map_size: list[int]
    ship_spawn: list[float]
    ship_heading: float
    scenarios: list[_ScenarioJsonDict]


class ScenarioConfig(TypedDict, total=False):
    profile_path: Path
    seed: int | None
    is_deterministic: bool


def load_scenario_suite(config: ScenarioConfig) -> list[Scenario]:
    scenarios: list[Scenario] = []

    file_path = config.get("profile_path")
    if not file_path or not file_path.is_file():
        raise FileNotFoundError(
            f"Scenario configuration sheet missing: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = cast(_RootJsonSchema, json.load(f))

    rng = np.random.default_rng(config.get("seed"))
    is_deterministic = config.get("is_deterministic", False)

    map_size = cast(tuple[int, int], tuple(data["map_size"]))
    ship_spawn = tuple(data["ship_spawn"])
    ship_heading = data["ship_heading"]

    for scene_data in data["scenarios"]:
        asteroid_states: list[
            dict[str, tuple[float, float] | float | int]
        ] = []

        for group in scene_data["asteroid_groups"]:
            count = group["count"]
            size = group["size"]
            speed_range = group["speed_range"]
            angle_range = group["angle_range"]
            zones = group["pos_zones"]

            if is_deterministic:
                speed = (speed_range[0] + speed_range[1]) / 2.0
                angle = (angle_range[0] + angle_range[1]) / 2.0
                zone_count = len(zones)

                for index in range(count):
                    zone = zones[index % zone_count]
                    pos_x = (zone["x_range"][0] + zone["x_range"][1]) / 2.0
                    pos_y = (zone["y_range"][0] + zone["y_range"][1]) / 2.0

                    asteroid_states.append(
                        {
                            "position": (pos_x, pos_y),
                            "speed": speed,
                            "angle": angle,
                            "size": size,
                        }
                    )
            else:
                speeds = rng.uniform(
                    speed_range[0],
                    speed_range[1],
                    size=count,
                )
                angles = rng.uniform(
                    angle_range[0],
                    angle_range[1],
                    size=count,
                )

                zone_count = len(zones)
                zone_indices = (
                    rng.integers(0, zone_count, size=count)
                    if zone_count > 1
                    else None
                )

                for i in range(count):
                    zone_index = (
                        cast(int, zone_indices[i])
                        if zone_indices is not None
                        else 0
                    )
                    zone = zones[zone_index]

                    x_range = zone["x_range"]
                    y_range = zone["y_range"]

                    pos_x = float(rng.uniform(x_range[0], x_range[1]))
                    pos_y = float(rng.uniform(y_range[0], y_range[1]))

                    asteroid_states.append(
                        {
                            "position": (pos_x, pos_y),
                            "speed": cast(float, speeds[i]),
                            "angle": cast(float, angles[i]),
                            "size": size,
                        }
                    )

        ammo_limit_multiplier = scene_data.get("ammo_limit_multiplier")
        resolved_ammo = (
            float(ammo_limit_multiplier)
            if ammo_limit_multiplier is not None
            else 0.0
        )

        scenarios.append(
            Scenario(
                name=scene_data["name"],
                asteroid_states=asteroid_states,
                ship_states=[
                    {"position": ship_spawn, "angle": ship_heading, "lives": 3}
                ],
                map_size=map_size,
                time_limit=scene_data["time_limit"],
                stop_if_no_ammo=scene_data["stop_if_no_ammo"],
                ammo_limit_multiplier=resolved_ammo,
            )
        )

    return scenarios
