def calculate_parallel_metrics(single_time, parallel_time, worker_count):
    speedup = single_time / parallel_time
    efficiency = speedup / worker_count

    return speedup, efficiency

