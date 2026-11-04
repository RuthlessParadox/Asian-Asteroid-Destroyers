import math


def norm_angle(angle: float) -> float:
    """Scales a standard Cartesian angle from (-180, 180] to a (-1.0, 1.0] universe."""
    return angle / 180.0


def norm_distance(distance: float, map_diagonal_length: float) -> float:
    return distance / map_diagonal_length


def norm_wrapped_distance(
    distance: float,
    map_width: float,
    map_height: float,
) -> float:
    max_wrapped_distance = 0.5 * math.sqrt(map_width**2 + map_height**2)
    return min(distance / max_wrapped_distance, 1.0)


def norm_speed_asteroid(speed: float) -> float:
    """Normalizes typical max asteroid tracking velocities against a 240.0 baseline."""
    return speed / 240.0


def norm_speed_ship(speed: float) -> float:
    return speed / 240.0


def norm_turn_rate_ship(turn_rate: float) -> float:
    return turn_rate / 180.0


def norm_tti(tti: float) -> float:
    return tti / 10.0


def norm_size(size: float) -> float:
    """Maps discrete asteroid scales (1 to 4) directly onto a (-0.5, 1.0] target space."""
    return size / 2.0 - 1.0


def norm_asteroid_count(num: float) -> float:
    """Normalizes dense asteroid field cluster counts against a baseline maximum of 100."""
    return num / 100.0


def norm_ammo_ratio(ratio: float) -> float:
    capped_ratio = min(ratio, 2.0)
    return capped_ratio / 2.0


def denorm_thrust(normalized_thrust: float) -> float:
    return normalized_thrust * 480.0


def denorm_turn_rate(normalized_turn_rate: float) -> float:
    return normalized_turn_rate * 180.0
