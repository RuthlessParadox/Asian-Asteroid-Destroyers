from collections.abc import Callable
from typing import NamedTuple

type FuzzySet = Callable[[int | float], int | float]

class FuzzyRule(NamedTuple):
    fuzzy_sets: tuple[FuzzySet, ...]
    output: int | float