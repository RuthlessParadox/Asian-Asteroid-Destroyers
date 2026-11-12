# Goal: make benchmarking repeatable to compare performances under different conditions

# compare training configurations

# Configurations: define population size, generations, scenarios, etc
# run_benchmarks: iterate through configurations
# executes one complete training experiment
# metrics, runtime, best fitness, average fitness, convergence, evaluation time

# Fixed population size
# Fixed generations
# Fixed scenarios
# Multiple random seeds
# Total random time
# Runtime per generation
# Final best fitness

import time


def benchmark_training(run_training, configurations):
    results = []
    for config in configurations:  # describe what conditions to train under
        # set of parameters that determines how an experiment runs
        start_time = time.perf_counter()
        training_results = run_training(config)  # training process to measure
        # receives a configuration
        # creates initial population
        # runs genetic algorithm
        # evaluates individuals using fuzzy controller and game scenarios
        # records performance metrics
        # returns results
        elapsed_time = time.perf_counter() - start_time
        results.append(
            {
                "configuration": config,
                "runtime_seconds": elapsed_time,
                "training_results": training_results[0],
            }
        )
    return results


# TODO create callbacks and data structures
# TODO data structure for storing benchmarking properties
# TODO single loop
