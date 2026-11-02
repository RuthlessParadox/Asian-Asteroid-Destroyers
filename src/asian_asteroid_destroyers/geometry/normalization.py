def norm_angle(angle: float) -> float:
    """Scales a standard Cartesian angle from (-180, 180] to a (-1.0, 1.0] universe."""
    return angle / 180.0


def norm_distance(distance: float, map_diagonal_length: float) -> float:
    """Normalizes a raw spatial distance vector against half the map's max diagonal."""
    return distance / (map_diagonal_length / 2.0)


def norm_speed_asteroid(speed: float) -> float:
    """Normalizes typical max asteroid tracking velocities against a 240.0 baseline."""
    return speed / 240.0


def norm_speed_ship(speed: float) -> float:
    """Normalizes typical ship velocities against a 240.0 baseline."""
    return speed / 240.0


def norm_tti(tti: float) -> float:
    """Scales impact windows relative to a critical 10-second response threshold."""
    return tti / 10.0


def norm_size(size: float) -> float:
    """Maps discrete asteroid scales (1 to 4) directly onto a (-0.5, 1.0] target space."""
    return size / 2.0 - 1.0


def norm_asteroid_count(num: float) -> float:
    """Normalizes dense asteroid field cluster counts against a baseline maximum of 100."""
    return num / 100.0
