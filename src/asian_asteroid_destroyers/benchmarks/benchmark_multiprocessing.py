# Compare training or fitness evaluation using different worker counts
# Measure worker scaling

# Single process baseline
# 2, 4, 8 workers
# Physical core worker count
# Logical processor worker count
# Speedup
# parallel efficieny


def calculate_parallel_metrics(single_time, parallel_time, worker_count):
    if parallel_time <= 0 or single_time < 0 or worker_count <= 0:
        raise ValueError()
    speedup = single_time / parallel_time
    efficiency = speedup / worker_count

    return speedup, efficiency
