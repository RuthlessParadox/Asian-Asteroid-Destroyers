class FuzzySet:
    def __init__(self, function):
        self.function = function

    def calculate(self, x):
        return self.function(x)

class FuzzyRule:
    def __init__(self, fuzzy_sets, output):
        self.fuzzy_sets = fuzzy_sets
        self.output = output