class FuzzySet:
    def __init__(self, function):
        self.function = function

    def calculate(self, x):
        return self.function(x)