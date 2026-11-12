# measure scenario evaluation cost

# Number of scenarios
# Repeats per scenario
# Total evaluation time
# Average time per scenario
# Average controller score
# Score variability across scenarios

import statistics
import time
from tracemalloc import start


def benchmark_scenarios(evaluate_controller, controller, scenarios):
    if not scenarios:
        return None

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
        "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
    }
