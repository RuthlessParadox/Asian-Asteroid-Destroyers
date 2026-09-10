from collections.abc import Callable
from typing import NamedTuple

# Function that takes a numeric input and returns a firing strength.
type FuzzySet = Callable[[int | float], int | float]

# Tuple containing a FuzzySet tuple and a numeric output.
class FuzzyRule(NamedTuple):
    fuzzy_sets: tuple[FuzzySet, ...]
    output: int | float

# Function that takes 2 numeric inputs and returns a numeric output.
# Used to model the intersection of 2 fuzzy sets.
# Common t-norms: minimum, product, Lukasiewicz
# Requirements:
# 1. If an input is 1, the other input should be returned.
# 2. If an input increases and the other input does not,
#    the result cannot decrease.
# 3. The order of inputs should not matter.
# 4. The order of function calls should not matter.
type TNorm = Callable[[int | float, int | float], int | float]

def takagi_sugeno_calculation(
    inputs: tuple[int | float, ...],
    ruleset: tuple[FuzzyRule, ...],
    t_norm: TNorm,
) -> int | float:
    """Uses the Takagi-Sugeno fuzzy inference system to calculate an output.

    Iterates over a ruleset. Calculates firing strength for each rule by
    evaluating each input to the corresponding fuzzy set's membership function
    and then modeling a logical AND using the t-norm function. Multiplies each
    rule's firing strength by the rule's output value. Returns the weighted
    average of the total rule outputs over the total firing strengths.

    :param inputs: The input value(s) to use.
    :param ruleset: The rules to use.
    :param t_norm: The t-norm function to use.
    :return: The calculated output.
    """
    # Start with total output and total firing strength 0.
    total_outputs = 0.0
    total_firing_strengths = 0.0
    # Iterate through each rule in the ruleset.
    for rule in ruleset:
        # Start with firing strength 1 since the t-norm function should not
        # alter the other input.
        firing_strength = 1
        # Iterate through each fuzzy set in the rule.
        for index, fuzzy_set in enumerate(rule.fuzzy_sets):
            # Apply the t-norm function to the current firing strength and the
            # membership of the corresponding input to the fuzzy set.
            firing_strength = t_norm(firing_strength, fuzzy_set(inputs[index]))
        # Multiply firing strength by the rule's output to get the output of the
        # rule with the given inputs. Add to the total output.
        total_outputs += firing_strength * rule.output
        # Add the firing strength to the total firing strength.
        total_firing_strengths += firing_strength
    # Return the weighted average.
    return total_outputs / total_firing_strengths