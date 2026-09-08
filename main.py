class Rule:
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents
        self.consequent = consequent

def main():
    rule_1 = Rule(["good", "fast"], 0.25)
    rule_2 = Rule(["poor", "slow"], 0.1)
