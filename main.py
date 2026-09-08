# Takagi Sugeno Fuzzy System

# if x is A(i) and y is B(k) then z = z(ik)

# output is
# summation (from (i = 1 to n; k = 1 to m)
# (A(i)(x) max (B)(k)(x) * z(ik)))
# divided by summation ((from (i = 1 to n; k = 1 to m))
# (A(i)(x) max (B)(k)(x) * z(ik)))

class Rule:
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents
        self.consequent = consequent

def main():
    rule_1 = Rule(["good", "fast"], 0.25)
    rule_2 = Rule(["poor", "slow"], 0.1)
