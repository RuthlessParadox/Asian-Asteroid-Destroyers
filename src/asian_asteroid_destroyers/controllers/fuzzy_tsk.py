import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def trimf_batch(
    x: Array,
    a: float,
    b: float,
    c: float,
) -> Array:
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    if a == b:
        result[x <= b] = 1.0
    else:
        rising = (x >= a) & (x <= b)
        result[rising] = (x[rising] - a) / (b - a)

    if b == c:
        result[x >= b] = 1.0
    else:
        failing = (x > b) & (x <= c)
        result[failing] = (c - x[failing]) / (c - b)

    return np.clip(result, 0, 1)


def memberships(
    x: Array,
    center: float,
    variable: str,
) -> Array:
    x = np.asarray(x, dtype=np.float64)
    if variable == "distance":
        triangles = (
            (0.0, 0.0, center),
            (0.0, center, 1.0),
            (center, 1.0, 1.0),
        )
    elif variable == "angle":
        triangles = (
            (-1.0, -1.0, center),
            (-1.0, center, 1.0),
            (center, -1.0, 1.0),
        )
    else:
        raise ValueError(f"Unknown variable: {variable}")
    return np.stack(
        [trimf_batch(x, *triangle) for triangle in triangles],
        axis=-1,
    )


def tsk_zero_order(
    angle: Array,
    distance: Array,
    angle_center: float,
    distance_center: float,
    consequents: Array,
) -> Array:
    angle = np.asarray(angle, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    consequents = np.asarray(consequents, dtype=np.float64)

    if consequents.shape != (9,):
        raise ValueError()
    angle_mf = memberships(angle, angle_center, "angle")
    distance_mf = memberships(distance, distance_center, "distance")

    activations = (
        angle_mf[..., :, np.newaxis] * distance_mf[..., np.newaxis, :]
    )

    activations = activations.reshape(activations.shape[:-2] + (9,))
    numerator = np.sum(activations * consequents, axis=-1)
    denominator = np.sum(activations, axis=-1)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )
