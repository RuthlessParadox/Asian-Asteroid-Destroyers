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

    def start_timer(self):
        return time.perf_counter()

    def stop_timer(self, start_time):
        return time.perf_counter() - start_time

    def record_generation(self, fitness_values, elapsed_time):
        self.generation_times.append(elapsed_time)
        self.best_fitness.append(max(fitness_values))
        self.average_fitness.append(statistics.mean(fitness_values))
        self.median_fitness.append(statistics.median(fitness_values))

        if len(fitness_values) < 1:
            self.fitness_std.append(statistics.stdev(fitness_values))
        else:
            self.fitness_std.append(0.0)

    def get_total_fitness(self):
        return time.perf_counter() - self.training_start
