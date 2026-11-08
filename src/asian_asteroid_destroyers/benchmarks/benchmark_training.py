import time

def benchmark_training(run_training, configurations):
    results = []
    for config in configurations:
        start_time = time.perf_counter()
        best_fitness = run_training(config)
        elapsed_time = time.perf_counter() - start_time
        results.append({
            "configuration": config,
            "runtime_seconds": elapsed_time,
            "best_fitness": best_fitness,
        })
    return results
