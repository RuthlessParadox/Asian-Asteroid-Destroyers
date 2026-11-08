import time

from asian_asteroid_destroyers.benchmarks.metrics import (
    BenchMarkMetrics,
)
from asian_asteroid_destroyers.evokit import population


def train(population, num_generations):

    metrics = BenchMarkMetrics()
    best_fitness = float("inf")
    stagnation_count = 0
    stagnation_limit = 20

    for generation in range(num_generations):
        generation_start = time.perf_counter()
        elapsed = time.perf_counter() - generation_start

        fitness_values = [
            individual.fitness.values[0]
            for individual in population
            if individual.fitness.valid
        ]

        if not fitness_values:
            continue
        metrics.record_generation(fitness_values, elapsed)

        current_best = max(ind.fitness.values for ind in population)
        if current_best > best_fitness:
            best_fitness = current_best
            stagnation_count = 0
        else:
            stagnation_count += 1
        if stagnation_count >= stagnation_limit:
            break
    return population, metrics
