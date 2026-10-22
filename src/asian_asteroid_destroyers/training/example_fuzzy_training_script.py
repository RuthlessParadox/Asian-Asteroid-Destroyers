# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import random
import sys
import time
from pathlib import Path

import polars as pl
from deap import base, creator, tools

sys.path.append("")
import json
import os

FitnessMax = getattr(creator, "FitnessMax", None)

if FitnessMax is None:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    FitnessMax = getattr(creator, "FitnessMax")

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=FitnessMax)

Individual = getattr(creator, "Individual")
from asian_asteroid_destroyers.training.data_storage import (
    GenerationData,
    add_generation,
    build_best_solutions_lazy_frame,
    build_timestamp,
    save_best_solutions,
)
from asian_asteroid_destroyers.training.fitness_functions import (
    example_fitness,
)

GENOME_LENGTH = 24
POPULATION_SIZE = 5
GENERATIONS = 3

# original fitness function from DEAP onemax example
# def evalOneMax(individual):
#     # print("test", individual[:2])
#     # print(type(individual))
#     return sum(individual),


def record_generation(
    best_solutions,
    best_individual,
    generation,
    population_size,
    cxpb,
    mutpb,
):
    data = GenerationData(
        generation=generation,
        fitness=float(best_individual.fitness.values[0]),
        genome=[float(gene) for gene in best_individual],
        number_of_genes=len(best_individual),
        population_size=population_size,
        cxpb=float(cxpb),
        mutpb=float(mutpb),
    )
    return add_generation(best_solutions, data)


def clear_solution_history(output_dir):
    """
    Remove any stale generation checkpoint JSON files from a previous run.

    :param output_dir: Directory containing best_solution_gen_XXXX.json files
    :return: None
    """
    if not output_dir.exists():
        return

    for path in output_dir.glob("best_solution_gen_*.json"):
        path.unlink()


def save_best_solution(
    best_individual, generation, output_dir, population_size, cxpb, mutpb
):
    """
    Save the current best solution in a human-readable JSON file.

    :param best_individual: DEAP individual containing the current best solution
    :param generation: Current generation number
    :param output_dir: Directory where generation JSON files will be written
    :param population_size: Number of individuals in the population
    :param cxpb: Crossover probability
    :param mutpb: Mutation probability
    :return: None
    """
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
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def normalize_storage_schema(best_solutions):
    return best_solutions.with_columns(
        pl.col("generation").cast(pl.Int64),
        pl.col("fitness").cast(pl.Float64),
        pl.col("genome").cast(pl.List(pl.Float64)),
        pl.col("number_of_genes").cast(pl.Int64),
        pl.col("population_size").cast(pl.Int64),
        pl.col("cxpb").cast(pl.Float64),
        pl.col("mutpb").cast(pl.Float64),
    )


def evaluate_individual(individual):
    start = time.perf_counter()
    print("Starting fitness evaluation...")

    fitness = example_fitness(individual)

    elapsed = time.perf_counter() - start
    print(f"Fitness evaluation finished in {elapsed:.2f}s")

    return fitness


def main():
    """
    Main function for GA-based training with DEAP package
    :return: None
    """
    CXPB, MUTPB = 0.5, 0.2
    CHECKPOINT_INTERVAL = 25
    timestamp = build_timestamp()

    best_solutions = build_best_solutions_lazy_frame()
    best_solutions = normalize_storage_schema(best_solutions)

    # creating individual and fitness characteristics/types

    toolbox = base.Toolbox()
    # Attribute generator
    # creating a float attribute - this will be the encoding of the GA individuals so all of the DNA in a chromosome will be floats
    # The random component is to initialize the population with random values
    # The random component is to initialize the population with random values
    toolbox.register("attr_flt1", random.uniform, 0.0, 1.0)
    attr_flt1 = getattr(toolbox, "attr_flt1")
    # Structure initializers
    # this registers an individual as an iterable of 50 values where each value is using the float attribute we just defined above
    toolbox.register(
        "individual",
        tools.initRepeat,
        Individual,
        attr_flt1,
        GENOME_LENGTH,
    )
    individual = getattr(toolbox, "individual")
    # this creates our population by creating a list of the individuals as defined above
    toolbox.register("population", tools.initRepeat, list, individual)
    population = getattr(toolbox, "population")

    # here we register our custom fitness function that we import above - in the DEAP onemax example this is where "evalOneMax" would be passed instead
    toolbox.register("evaluate", evaluate_individual)
    # Defining what type of crossover will be used
    toolbox.register("mate", tools.cxTwoPoint)
    # defining what type of mutation will be used - in this case our encoding is floats so we're using Gaussian - we could use other methods
    toolbox.register(
        "mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05
    )
    # defining how selection is performed - we use tournament selection like the DEAP onemax example
    toolbox.register("select", tools.selTournament, tournsize=3)

    mate = getattr(toolbox, "mate")
    mutate = getattr(toolbox, "mutate")
    select = getattr(toolbox, "select")
    clone = getattr(toolbox, "clone")
    # creating a population - in this case we only have 20 individuals in our population
    pop = population(n=POPULATION_SIZE)

    # Path for saving the best solution checkpoint each generation
    best_solution_dir = Path(
        os.path.dirname(__file__), "../../../training_data/solution_history"
    )
    clear_solution_history(best_solution_dir)

    # Evaluate the entire population
    fitnesses = list(map(evaluate_individual, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Use DEAP's standard HallOfFame utility to track the best individual found so far
    hof = tools.HallOfFame(1)
    hof.update(pop)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Save the best initial solution before evolution begins
    # save_best_solution(hof[0], g, best_solution_dir, len(pop), CXPB, MUTPB)

    best_solutions = add_generation(
        best_solutions,
        GenerationData(
            g,
            hof[0].fitness.values[0],
            list(hof[0]),
            len(hof[0]),
            len(pop),
            CXPB,
            MUTPB,
        ),
    )

    # Begin the evolution
    while g < GENERATIONS:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                mutate(mutant)
                # clamp values after mutation to allowable range (0, 1)
                for i in range(len(mutant)):
                    if mutant[i] < 0.0:
                        mutant[i] = 0.0
                    elif mutant[i] > 1.0:
                        mutant[i] = 1.0
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(evaluate_individual, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Update the best-so-far individual and save it every generation
        hof.update(pop)
        best_solutions = add_generation(
            best_solutions,
            GenerationData(
                g,
                hof[0].fitness.values[0],
                list(hof[0]),
                len(hof[0]),
                len(pop),
                CXPB,
                MUTPB,
            ),
        )

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean**2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print("  Best so far %s" % (hof[0].fitness.values[0],))

    hof.update(pop)

    print("\nFinal best individual:")
    print(hof[0])
    print("Final best fitness:", hof[0].fitness.values)
    print(f"Best solution history saved to {best_solution_dir.resolve()}")

    save_best_solutions(best_solutions, timestamp)


if __name__ == "__main__":
    main()
