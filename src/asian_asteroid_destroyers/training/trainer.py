import copy
import json
import random
from pathlib import Path

from deap import base, creator, tools

from asian_asteroid_destroyers.training import config
from asian_asteroid_destroyers.training.evaluation import (
    evaluate_individual,
)
from asian_asteroid_destroyers.training.fitness_functions import (
    create_fitness_function,
)

FITNESS_WEIGHTS = {
    "fraction_total_asteroids_hit": 1.0,
    "accuracy": 1.0,
    "deaths": -0.2,
}


def initialize_deap_types():
    """Initialize DEAP fitness and individual types."""
    fitness_type = getattr(creator, "FitnessMax", None)

    if fitness_type is None:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        fitness_type = getattr(creator, "FitnessMax")

    if not hasattr(creator, "Individual"):
        creator.create(
            "Individual",
            list,
            fitness=fitness_type,
        )

    return getattr(creator, "Individual")


def create_toolbox():
    """Register the genetic algorithm operators."""
    fitness_function = create_fitness_function(config.FITNESS_WEIGHTS)
    individual_type = initialize_deap_types()
    toolbox = base.Toolbox()

    toolbox.register("attr_flt1", random.uniform, 0.0, 1.0)
    attr_flt1 = getattr(toolbox, "attr_flt1")

    toolbox.register(
        "individual",
        tools.initRepeat,
        individual_type,
        attr_flt1,
        n=config.GENOME_LENGTH,
    )
    individual = getattr(toolbox, "individual")

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        individual,
    )

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=config.TOURNAMENT_SIZE,
    )
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=0.0,
        sigma=config.MUTATION_SIGMA,
        indpb=config.MUTATION_INDIVIDUAL_PROBABILITY,
    )

    return toolbox


def create_population():
    """Create the initial population."""
    random.seed(config.RANDOM_SEED)
    toolbox = create_toolbox()
    population = getattr(toolbox, "population")
    pop = population(n=config.POPULATION_SIZE)

    return toolbox, pop


def clear_solution_history(output_dir):
    """Remove stale generation checkpoint JSON files."""
    if not output_dir.exists():
        return

    for path in output_dir.glob("best_solution_gen_*.json"):
        path.unlink()


def save_best_solution(
    best_individual,
    generation,
    output_dir,
    population_size,
    cxpb,
    mutpb,
):
    """Save the best individual for a generation."""
    result = {
        "generation": generation,
        "fitness": list(best_individual.fitness.values),
        "genome": list(best_individual),
        "n_genes": len(best_individual),
        "population_size": population_size,
        "cxpb": cxpb,
        "mutpb": mutpb,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"best_solution_gen_{generation:04d}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def run_training():
    """Run the genetic algorithm training process."""
    toolbox, pop = create_population()

    evaluate = getattr(toolbox, "evaluate")
    select = getattr(toolbox, "select")
    clone = getattr(toolbox, "clone")
    mate = getattr(toolbox, "mate")
    mutate = getattr(toolbox, "mutate")

    cxpb = config.CROSSOVER_PROBABILITY
    mutpb = config.MUTATION_PROBABILITY

    # Save generation checkpoints.
    best_solution_dir = (
        Path(__file__).resolve().parent
        / "../../../training_data/solution_history"
    ).resolve()

    clear_solution_history(best_solution_dir)

    # Evaluate the initial population.
    fitnesses = list(map(evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Track the best individual.
    hof = tools.HallOfFame(1)
    hof.update(pop)

    generation = 0
    save_best_solution(
        hof[0], generation, best_solution_dir, len(pop), cxpb, mutpb
    )

    # Evolve the population.
    while generation < config.GENERATIONS:
        generation += 1
        print(f"-- Generation {generation} --")

        offspring = select(pop, len(pop))
        offspring = list(map(clone, offspring))

        # Crossover.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation.
        for mutant in offspring:
            if random.random() < mutpb:
                mutate(mutant)

                # Keep every gene within [0, 1].
                for i in range(len(mutant)):
                    mutant[i] = max(0.0, min(1.0, mutant[i]))

                del mutant.fitness.values

        # Evaluate individuals whose fitness changed.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Update and save the best solution.
        hof.update(pop)
        save_best_solution(
            hof[0], generation, best_solution_dir, len(pop), cxpb, mutpb
        )

        # Print generation statistics.
        fits = [ind.fitness.values[0] for ind in pop]
        avg = sum(fits) / len(fits)
        sum2 = sum(value * value for value in fits)
        std = abs(sum2 / len(fits) - avg**2) ** 0.5

        print(f"  Min {min(fits)}")
        print(f"  Max {max(fits)}")
        print(f"  Avg {avg}")
        print(f"  Std {std}")
        print(f"  Best so far {hof[0].fitness.values[0]}")

    print("\nFinal best individual:")
    print(hof[0])
    print("Final best fitness:", hof[0].fitness.values)
    print(f"Best solution history saved to {best_solution_dir}")
