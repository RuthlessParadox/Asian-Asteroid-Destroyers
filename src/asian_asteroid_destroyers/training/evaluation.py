import time

from asian_asteroid_destroyers.training.fitness_functions import (
    create_fitness_function,
)


def evaluate_individual(individual, fitness_function=None):
    """Evaluate an individual and report its execution time."""
    if fitness_function is None:
        fitness_function = create_fitness_function()

    start = time.perf_counter()
    print("Evaluating individual", individual)

    fitness = fitness_function(individual)

    elapsed = time.perf_counter() - start
    print(f"Evaluation time: {elapsed:.2f}s")

    return fitness
