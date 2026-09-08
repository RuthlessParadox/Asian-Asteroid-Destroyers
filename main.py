# Takagi Sugeno Fuzzy System

# if x is A(i) and y is B(k) then z = z(ik)

# output is
# summation (from (i = 1 to n; k = 1 to m)
# (A(i)(x) max (B)(k)(x) * z(ik)))
# divided by summation ((from (i = 1 to n; k = 1 to m))
# (A(i)(x) max (B)(k)(x) * z(ik)))

def TakagiSugeno(list1, list2):
    numerator = 0
    denominator = 0
    result = 0
    for i in range (len(list1)) :
        numerator += list1[i] * list2[i]
        denominator += list1[i]
    result = numerator / denominator

class Rule:
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents
        self.consequent = consequent

def main():
    rule_1 = Rule(["good", "fast"], 0.25)
    rule_2 = Rule(["poor", "slow"], 0.1)
