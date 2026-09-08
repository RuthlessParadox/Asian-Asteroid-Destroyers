class FuzzySet:
    def __init__(self, functions):
        self.functions = functions

    def calculate(self, x):
        return tuple(function(x) for function in self.functions)