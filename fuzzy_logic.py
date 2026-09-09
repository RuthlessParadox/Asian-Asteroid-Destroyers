from collections.abc import Callable
from typing import NamedTuple

type FuzzySet = Callable[[int | float], int | float]

class FuzzyRule(NamedTuple):
    fuzzy_sets: tuple[FuzzySet, ...]
    output: int | float

class TNorm:
    def __init__(self, function):
        self.function = function

    def calculate(self, a, b):
        return self.function(a, b)

def takagi_sugeno_calculation(inputs, ruleset, t_norm):
    total_outputs = 0.0
    total_firing_strengths = 0.0
    for rule in ruleset:
        firing_strength = 1
        for index, fuzzy_set in rule.fuzzy_sets:
            firing_strength = t_norm.calculate(firing_strength, inputs[index])
        total_outputs += firing_strength * rule.output
        total_firing_strengths += firing_strength
    return total_outputs / total_firing_strengths