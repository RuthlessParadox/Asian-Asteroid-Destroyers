# Generations of best fitness
# Number of generations till target fitness
# Number of generations without improvement
# Final best and average fitness
# Percentage of runs that coverage

# Training Runtime
# Generation Runtime
# Evaluation Runtime
# Individuals Evaluated Per Second
# CPU and Memory Usage
# Best, Average, Median Fitness
# Stagnation Count
# Scenario Gameplay Metrics

# should only collect data; not control training

import statistics
import time


class BenchMarkMetrics:
    def __init__(self):
        self.training_start = time.perf_counter()
        self.generation_times = []
        self.evaluation_times = []
        self.best_fitness = []
        self.average_fitness = []
        self.median_fitness = []
        self.fitness_std = []

    # ===========================
    # Training Runtime
    # ===========================
    def start_timer(self):
        return time.perf_counter()

    def stop_timer(self, start_time):
        return time.perf_counter() - start_time

    # ===========================
    # Generation Runtime
    # ===========================
    def record_generation(self, fitness_values, elapsed_time):
        if not fitness_values:
            return
        self.generation_times.append(elapsed_time)
        self.best_fitness.append(max(fitness_values))
        self.average_fitness.append(statistics.mean(fitness_values))
        self.median_fitness.append(statistics.median(fitness_values))

        std = (
            statistics.stdev(fitness_values)
            if len(fitness_values) > 1
            else 0.0
        )

        self.fitness_std.append(std)

    # ===========================
    # Fitness Standard Deviation
    # ===========================
    def get_total_runtime(self):
        return time.perf_counter() - self.training_start
