# Atreyi Seal, Isaac Zhou
# CSP 200
#
# Simple application of fuzzy logic modeling tipping for a restaurant.

import sys
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import fuzzy_logic

@overload
def food_quality_poor(x: float) -> int | float: ...
@overload
def food_quality_poor(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def food_quality_poor(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 0 <= x <= 5:
            return (5 - x) / 5
        else:
            return 0
    else:
        return np.where(np.logical_and(0 <= x, x <= 5), (5 - x) / 5, 0)

@overload
def food_quality_average(x: float) -> int | float: ...
@overload
def food_quality_average(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def food_quality_average(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 2.5 <= x <= 5:
            return (x - 2.5) / 2.5
        elif 5 < x <= 7.5:
            return (7.5 - x) / 2.5
        else:
            return 0
    else:
        conditions = [
            np.logical_and(2.5 <= x, x <= 5),
            np.logical_and(5 < x, x <= 7.5)
        ]
        choices = [
            (x - 2.5) / 2.5,
            (7.5 - x) / 2.5
        ]
        return np.select(conditions, choices, default=0)

@overload
def food_quality_excellent(x: float) -> int | float: ...
@overload
def food_quality_excellent(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def food_quality_excellent(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 5 <= x <= 10:
            return (x - 5) / 5
        else:
            return 0
    else:
        return np.where(np.logical_and(5 <= x, x <= 10), (x - 5) / 5, 0)

@overload
def service_poor(x: float) -> int | float: ...
@overload
def service_poor(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def service_poor(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 0 <= x <= 5:
            return (5 - x) / 5
        else:
            return 0
    else:
        return np.where(np.logical_and(0 <= x, x <= 5), (5 - x) / 5, 0)

@overload
def service_average(x: float) -> int | float: ...
@overload
def service_average(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def service_average(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 2.5 <= x <= 5:
            return (x - 2.5) / 2.5
        elif 5 < x <= 7.5:
            return (7.5 - x) / 2.5
        else:
            return 0
    else:
        conditions = [
            np.logical_and(2.5 <= x, x <= 5),
            np.logical_and(5 < x, x <= 7.5)
        ]
        choices = [
            (x - 2.5) / 2.5,
            (7.5 - x) / 2.5
        ]
        return np.select(conditions, choices, default=0)

@overload
def service_excellent(x: float) -> int | float: ...
@overload
def service_excellent(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def service_excellent(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 5 <= x <= 10:
            return (x - 5) / 5
        else:
            return 0
    else:
        return np.where(np.logical_and(5 <= x, x <= 10), (x - 5) / 5, 0)

@overload
def timing_slow(x: float) -> int | float: ...
@overload
def timing_slow(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def timing_slow(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 0 <= x <= 20:
            return 0
        elif 20 < x <= 30:
            return (x - 20) / 10
        else:
            return 1
    else:
        conditions = [
            np.logical_and(0 <= x, x <= 20),
            np.logical_and(20 < x, x <= 30)
        ]
        choices = [
            0,
            (x - 20) / 10
        ]
        return np.select(conditions, choices, default=1)

@overload
def timing_medium(x: float) -> int | float: ...
@overload
def timing_medium(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def timing_medium(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 5 <= x <= 15:
            return (x - 5) / 10
        elif 15 < x <= 25:
            return (25 - x) / 10
        else:
            return 0
    else:
        conditions = [
            np.logical_and(5 <= x, x <= 15),
            np.logical_and(15 < x, x <= 25)
        ]
        choices = [
            (x - 5) / 10,
            (25 - x) / 10
        ]
        return np.select(conditions, choices, default=0)

@overload
def timing_fast(x: float) -> int | float: ...
@overload
def timing_fast(
    x: npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> npt.NDArray[np.int_] | npt.NDArray[np.floating]: ...
def timing_fast(
    x: float | npt.NDArray[np.int_] | npt.NDArray[np.floating]
) -> int | float | npt.NDArray[np.int_] | npt.NDArray[np.floating]:
    if isinstance(x, (int, float)):
        if 0 <= x <= 10:
            return (10 - x) / 10
        else:
            return 0
    else:
        return np.where(np.logical_and(0 <= x, x <= 10), (10 - x) / 10, 0)

def product(a: float, b: float) -> int | float:
    return a * b

def main() -> int:
    food_quality = (
        food_quality_poor,
        food_quality_average,
        food_quality_excellent
    )
    service = (
        service_poor,
        service_average,
        service_excellent
    )
    timing = (
        timing_slow,
        timing_medium,
        timing_fast
    )

    ruleset = (
        fuzzy_logic.FuzzyRule((food_quality[0], service[0], timing[0]), 0),
        fuzzy_logic.FuzzyRule((food_quality[0], service[0], timing[1]), 0.05),
        fuzzy_logic.FuzzyRule((food_quality[0], service[0], timing[2]), 0.075),
        fuzzy_logic.FuzzyRule((food_quality[0], service[1], timing[0]), 0.05),
        fuzzy_logic.FuzzyRule((food_quality[0], service[1], timing[1]), 0.075),
        fuzzy_logic.FuzzyRule((food_quality[0], service[1], timing[2]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[0], service[2], timing[0]), 0.075),
        fuzzy_logic.FuzzyRule((food_quality[0], service[2], timing[1]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[0], service[2], timing[2]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[1], service[0], timing[0]), 0.05),
        fuzzy_logic.FuzzyRule((food_quality[1], service[0], timing[1]), 0.075),
        fuzzy_logic.FuzzyRule((food_quality[1], service[0], timing[2]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[1], service[1], timing[0]), 0.075),
        fuzzy_logic.FuzzyRule((food_quality[1], service[1], timing[1]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[1], service[1], timing[2]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[1], service[2], timing[0]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[1], service[2], timing[1]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[1], service[2], timing[2]), 0.2),
        fuzzy_logic.FuzzyRule((food_quality[2], service[0], timing[0]), 0.075),
        fuzzy_logic.FuzzyRule((food_quality[2], service[0], timing[1]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[2], service[0], timing[2]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[2], service[1], timing[0]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[2], service[1], timing[1]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[2], service[1], timing[2]), 0.2),
        fuzzy_logic.FuzzyRule((food_quality[2], service[2], timing[0]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[2], service[2], timing[1]), 0.2),
        fuzzy_logic.FuzzyRule((food_quality[2], service[2], timing[2]), 0.25),

    )

    print("T-Norm: Minimum\n")
    print("Food Quality: 7, Service: 3, Time: 10")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((7, 3, 10), ruleset, min)}")
    print("Food Quality: 5, Service: 2, Time: 7")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((5, 2, 7), ruleset, min)}")
    print("Food Quality: 3, Service: 10, Time: 34")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((3, 10, 34), ruleset, min)}")
    print("Food Quality: 6, Service: 6, Time: 14")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((6, 6, 14), ruleset, min)}")
    print("Food Quality: 8, Service: 2, Time: 2")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((8, 2, 2), ruleset, min)}")
    print("\n")

    print("T-Norm: Product\n")
    print("Food Quality: 7, Service: 3, Time: 10")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((7, 3, 10), ruleset, product)}")
    print("Food Quality: 5, Service: 2, Time: 7")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((5, 2, 7), ruleset, product)}")
    print("Food Quality: 3, Service: 10, Time: 34")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((3, 10, 34), ruleset, product)}")
    print("Food Quality: 6, Service: 6, Time: 14")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((6, 6, 14), ruleset, product)}")
    print("Food Quality: 8, Service: 2, Time: 2")
    print(f"Tip: {fuzzy_logic.takagi_sugeno_calculation((8, 2, 2), ruleset, product)}")

    fig_food_quality, ax_food_quality = plt.subplots()
    ax_food_quality.set_title("Food Quality Membership Functions")
    ax_food_quality.set_xlabel("Food Quality")
    ax_food_quality.set_ylabel("Membership")
    x = np.linspace(0, 10, 100)
    ax_food_quality.plot(x, food_quality_poor(x), label="Poor")
    ax_food_quality.plot(x, food_quality_average(x), label="Average")
    ax_food_quality.plot(x, food_quality_excellent(x), label="Excellent")
    box = ax_food_quality.get_position()
    ax_food_quality.set_position((box.x0, box.y0, box.width * 0.8, box.height))

    fig_service, ax_service = plt.subplots()
    ax_service.set_title("Service Membership Functions")
    ax_service.set_xlabel("Service")
    ax_service.set_ylabel("Membership")
    x = np.linspace(0, 10, 100)
    ax_service.plot(x, service_poor(x), label="Poor")
    ax_service.plot(x, service_average(x), label="Average")
    ax_service.plot(x, service_excellent(x), label="Excellent")
    box = ax_service.get_position()
    ax_service.set_position((box.x0, box.y0, box.width * 0.8, box.height))

    fig_timing, ax_timing = plt.subplots()
    ax_timing.set_title("Time Membership Functions")
    ax_timing.set_xlabel("Time (min)")
    ax_timing.set_ylabel("Membership")
    x = np.linspace(0, 30, 100)
    ax_timing.plot(x, timing_fast(x), label="Fast")
    ax_timing.plot(x, timing_medium(x), label="Medium")
    ax_timing.plot(x, timing_slow(x), label="Slow")
    box = ax_timing.get_position()
    ax_timing.set_position((box.x0, box.y0, box.width * 0.8, box.height))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    return 0

if __name__ == "__main__":
    sys.exit(main())