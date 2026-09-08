# Takagi Sugeno Fuzzy System

# if x is A(i) and y is B(k) then z = z(ik)

# output is
# summation (from (i = 1 to n; k = 1 to m)
# (A(i)(x) max (B)(k)(x) * z(ik)))
# divided by summation ((from (i = 1 to n; k = 1 to m))
# (A(i)(x) max (B)(k)(x) * z(ik)))

import fuzzy_logic

def food_quality_low(x):
    if 1 <= x <= 3:
        return (x - 1) / 2
    elif 3 < x <= 5:
        return (5 - x) / 2
    else:
        return 0

def main():
    fuzzy_set_food_quality_low = fuzzy_logic.FuzzySet(food_quality_low)