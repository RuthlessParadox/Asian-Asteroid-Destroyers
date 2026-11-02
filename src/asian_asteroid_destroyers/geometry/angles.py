def trim_angle(angle: float) -> float:
    """Clamps an arbitrary angle to the standard range of (-180.0, 180.0]."""
    return float((angle + 180.0) % 360.0 - 180.0)


def calculate_turn_rate(
    current_heading: float,
    target_heading: float,
) -> float:
    """Calculates the optimal timestep turn rate required to achieve a heading goal."""
    max_turn_rate_timestep = 6.0
    desired_diff = trim_angle(target_heading - current_heading)

    if desired_diff > max_turn_rate_timestep:
        return 180.0
    if desired_diff < -max_turn_rate_timestep:
        return -180.0
    return desired_diff * 30.0
