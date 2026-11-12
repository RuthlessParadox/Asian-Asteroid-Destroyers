# Analyze training progress

# Best fitness per generation
# Average fitness per generation
# Generations until target fitness
# Stagnation duration
# Final fitness
# Results across multiple random seeds


def generations_to_target(best_fitness_history, target_fitness):
    for generation, fitness in enumerate(best_fitness_history):
        if fitness >= target_fitness:
            return generation
    return None


def calculate_fitness(best_fitness_history, target_fitness):
    if not best_fitness_history:
        return 0
    best_so_far = best_fitness_history[0]
    stagnation = 0

    for fitness in best_fitness_history:
        if fitness > best_so_far:
            best_so_far = fitness
            stagnation = 0
        else:
            stagnation += 1
    return stagnation
