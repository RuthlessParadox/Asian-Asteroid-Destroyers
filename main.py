# Takagi Sugeno Fuzzy System

# if x is A(i) and y is B(k) then z = z(ik)

# output is
# summation (from (i = 1 to n; k = 1 to m)
# (A(i)(x) min (B)(k)(x) * z(ik)))
# divided by summation ((from (i = 1 to n; k = 1 to m))
# (A(i)(x) min (B)(k)(x))

import sys

import fuzzy_logic

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

    ruleset: tuple[fuzzy_logic.FuzzyRule, ...] = (
        fuzzy_logic.FuzzyRule((food_quality[0], service[0]), 0),
        fuzzy_logic.FuzzyRule((food_quality[0], service[1]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[0], service[2]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[1], service[0]), 0.1),
        fuzzy_logic.FuzzyRule((food_quality[1], service[1]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[1], service[2]), 0.2),
        fuzzy_logic.FuzzyRule((food_quality[2], service[0]), 0.15),
        fuzzy_logic.FuzzyRule((food_quality[2], service[1]), 0.2),
        fuzzy_logic.FuzzyRule((food_quality[2], service[2]), 0.25),
    )

    return 0

if __name__ == "__main__":
    sys.exit(main())

def food_quality_poor(x: int | float) -> int | float:
    if 0 <= x <= 2.5:
        return x / 2.5
    elif 2.5 < x <= 5:
        return (5 - x) / 2.5
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
    if 5 <= x <= 7.5:
        return (x - 5) / 2.5
    elif 7.5 < x <= 10:
        return (5 - x) / 2
    else:
        return 0

def service_poor(x: int | float) -> int | float:
    if 0 <= x <= 2.5:
        return x / 2.5
    elif 2.5 < x <= 5:
        return (5 - x) / 2.5
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
    if 5 <= x <= 7.5:
        return (x - 5) / 2.5
    elif 7.5 < x <= 10:
        return (5 - x) / 2
    else:
        return 0