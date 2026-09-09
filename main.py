# Takagi Sugeno Fuzzy System

# if x is A(i) and y is B(k) then z = z(ik)

# output is
# summation (from (i = 1 to n; k = 1 to m)
# (A(i)(x) max (B)(k)(x) * z(ik)))
# divided by summation ((from (i = 1 to n; k = 1 to m))
# (A(i)(x) max (B)(k)(x) * z(ik)))

import fuzzy_logic

def food_quality_poor(x):
    if 0 <= x <= 2.5:
        return x / 2.5
    elif 2.5 < x <= 5:
        return (5 - x) / 2.5
    else:
        return 0

def food_quality_average(x):
    if 2.5 <= x <= 5:
        return (x - 2.5)/ 2.5
    elif 5 < x <= 7.5:
        return (7.5 - x) / 2.5
    else:
        return 0

def food_quality_excellent(x):
    if 5 <= x <= 7.5:
        return (x - 5) / 2.5
    elif 7.5 < x <= 10:
        return (5 - x) / 2
    else:
        return 0

def service_poor(x):
    if 0 <= x <= 2.5:
        return x / 2.5
    elif 2.5 < x <= 5:
        return (5 - x) / 2.5
    else:
        return 0

def service_average(x):
    if 2.5 <= x <= 5:
        return (x - 2.5)/ 2.5
    elif 5 < x <= 7.5:
        return (7.5 - x) / 2.5
    else:
        return 0

def service_excellent(x):
    if 5 <= x <= 7.5:
        return (x - 5) / 2.5
    elif 7.5 < x <= 10:
        return (5 - x) / 2
    else:
        return 0

def main():
    food_quality = [
        fuzzy_logic.FuzzySet(food_quality_poor),
        fuzzy_logic.FuzzySet(food_quality_average),
        fuzzy_logic.FuzzySet(food_quality_excellent)
    ]
    service = [
        fuzzy_logic.FuzzySet(service_poor),
        fuzzy_logic.FuzzySet(service_average),
        fuzzy_logic.FuzzySet(service_excellent)
    ]

    ruleset = [
        fuzzy_logic.FuzzyRule([food_quality[0], service[0]], 0),
        fuzzy_logic.FuzzyRule([food_quality[0], service[1]], 0.1),
        fuzzy_logic.FuzzyRule([food_quality[0], service[2]], 0.15),
        fuzzy_logic.FuzzyRule([food_quality[1], service[0]], 0.1),
        fuzzy_logic.FuzzyRule([food_quality[1], service[1]], 0.15),
        fuzzy_logic.FuzzyRule([food_quality[1], service[2]], 0.2),
        fuzzy_logic.FuzzyRule([food_quality[2], service[0]], 0.15),
        fuzzy_logic.FuzzyRule([food_quality[2], service[1]], 0.2),
        fuzzy_logic.FuzzyRule([food_quality[2], service[2]], 0.25),
    ]