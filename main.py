# Takagi Sugeno Fuzzy System

# if x is A(i) and y is B(k) then z = z(ik)

# output is
# summation (from (i = 1 to n; k = 1 to m)
# (A(i)(x) min (B)(k)(x) * z(ik)))
# divided by summation ((from (i = 1 to n; k = 1 to m))
# (A(i)(x) min (B)(k)(x))

import sys

import matplotlib.pyplot as plt
import numpy as np

import fuzzy_logic

def TakagiSugeno(list1, list2):
    numerator = 0
    denominator = 0
    result = 0
    for i in range (len(list1)) :
        numerator += list1[i] * list2[i]
        denominator += list1[i]
    result = numerator / denominator

def food_quality_poor(x: int | float) -> int | float:
    if 0 <= x <= 5:
        return (5 - x) / 5
    else:
        return 0

def food_quality_average(x: int | float) -> int | float:
    if 2.5 <= x <= 5:
        return (x - 2.5)/ 2.5
    elif 5 < x <= 7.5:
        return (7.5 - x) / 2.5
    else:
        return 0

def food_quality_excellent(x: int | float) -> int | float:
    if 5 <= x <= 10:
        return (x - 5) / 5
    else:
        return 0

def service_poor(x: int | float) -> int | float:
    if 0 <= x <= 5:
        return (5 - x) / 5
    else:
        return 0

def service_average(x: int | float) -> int | float:
    if 2.5 <= x <= 5:
        return (x - 2.5)/ 2.5
    elif 5 < x <= 7.5:
        return (7.5 - x) / 2.5
    else:
        return 0

def service_excellent(x: int | float) -> int | float:
    if 5 <= x <= 10:
        return (x - 5) / 5
    else:
        return 0

def timing_slow(x: int | float) -> int | float:
    if 20 <= x <= 30:
        return (x - 20) / 10
    elif x <= 20:
        return 0
    else:
        return 1
    return 0
def timing_medium(x: int | float) -> int | float:
    if 5 <= x <= 15:
        return (x - 5) / 10
    elif 15 < x <= 25:
        return (25 - x) / 10
    else:
        return 0
def timing_fast(x: int | float) -> int | float:
    if 0 <= x <= 10:
        return (10 - x) / 10
    else:
        return 0

def product(a, b):
    return a * b

def main() -> int:
    food_quality: tuple[fuzzy_logic.FuzzySet, ...] = (
        food_quality_poor,
        food_quality_average,
        food_quality_excellent
    )
    service: tuple[fuzzy_logic.FuzzySet, ...] = (
        service_poor,
        service_average,
        service_excellent
    )
    timing: tuple[fuzzy_logic.FuzzySet, ...] = (
        timing_slow,
        timing_medium,
        timing_fast
    )

    ruleset: tuple[fuzzy_logic.FuzzyRule, ...] = (
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

    return 0

if __name__ == "__main__":
    sys.exit(main())