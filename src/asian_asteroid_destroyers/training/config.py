"""Configuration for fuzzy-controller genetic training."""

# Chromosome / controller
GENOME_LENGTH = 24

# Genetic algorithm
POPULATION_SIZE = 5
GENERATIONS = 3
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.2
MUTATION_SIGMA = 0.2
MUTATION_INDIVIDUAL_PROBABILITY = 0.05
TOURNAMENT_SIZE = 3

# Reproducibility. Set to None for a different random sequence each run.
RANDOM_SEED = 42
