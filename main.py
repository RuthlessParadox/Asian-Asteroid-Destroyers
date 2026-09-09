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
    food_quality = fuzzy_logic.FuzzySet(
        [food_quality_poor, food_quality_average, food_quality_excellent]
    )
    service = fuzzy_logic.FuzzySet(
        [service_poor, service_average, service_excellent]
    )