def generations_to_target(best_fitness_history, target_fitness):
    for generation, fitness in enumerate(best_fitness_history):
        if fitness == target_fitness:
            return generation
    return None
