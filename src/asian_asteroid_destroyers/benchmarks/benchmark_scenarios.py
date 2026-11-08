import time
import statistics
from tracemalloc import start

def benchmark_scenarios(evaluate_controller, controller, scenarios):
    scores = []
    start_time = time.perf_counter()
    for scenario in scenarios:
        score = evaluate_controller(controller, scenario)
        scores.append(score)

    elapsed_time = time.perf_counter() - start_time
    return {
        "scenario_count": len(scenarios),
        "total_time": elapsed_time,
        "average_runtime_per_scenario": elapsed_time / len(scenarios),
        "average_score": statistics.mean(scores),
        "score_std": statistics.stdev(scores)
        if len(scores) > 1
        else 0,
    }