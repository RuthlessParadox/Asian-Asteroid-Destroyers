from collections.abc import Callable
from typing import NamedTuple

type FuzzySet = Callable[[int | float], int | float]

class FuzzyRule(NamedTuple):
    fuzzy_sets: tuple[FuzzySet, ...]
    output: int | float

type TNorm = Callable[[int | float, int | float], int | float]

def takagi_sugeno_calculation(
    inputs: tuple[int | float, ...],
    ruleset: tuple[FuzzyRule, ...],
    t_norm: TNorm,
) -> int | float:
    total_outputs = 0.0
    total_firing_strengths = 0.0
    for rule in ruleset:
        firing_strength = 1
        for index, fuzzy_set in enumerate(rule.fuzzy_sets):
            firing_strength = t_norm(firing_strength, fuzzy_set(inputs[index]))
        total_outputs += firing_strength * rule.output
        total_firing_strengths += firing_strength
    return total_outputs / total_firing_strengths