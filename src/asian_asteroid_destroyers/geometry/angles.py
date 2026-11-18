def trim_angle(angle: float) -> float:
    """Clamps an arbitrary angle to the standard range of (-180.0, 180.0]."""
    return float((angle + 180.0) % 360.0 - 180.0)
