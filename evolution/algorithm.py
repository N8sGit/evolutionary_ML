from .population import initialize_population, population_diversity
from training import train_model
from .fitness import evaluate_fitness
from .selection import select_parents, crossover_models
from .mutation import mutate_model, adaptive_mutation_rate
from .speciate import speciate_population

import numpy as np
import torch
from data import train_loader, val_loader
import matplotlib.pyplot as plt
import random

def evolutionary_algorithm(pop_size=10, generations=5, base_mutation_rate=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population = initialize_population(pop_size)
    
    fitness_history = []
    diversity_history = []

    best_fitness_overall = -float('inf')
    best_model_overall = None

    initial_threshold = 0.5
    final_threshold = 0.1

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}")
        fitnesses = []
        f1s = []
        avg_losses = []
        
        # Train and evaluate each model in the population
        for i, model in enumerate(population):
            print(f"Training Model {i + 1}/{len(population)}")
            train_model(model, device, train_loader, epochs=1)
            fitness, f1, avg_loss = evaluate_fitness(model, device, val_loader, population)
            fitnesses.append(fitness)
            f1s.append(f1)
            avg_losses.append(avg_loss)
            print(f"Model {i + 1} - Fitness: {fitness:.4f}, F1 score: {f1:.4f}, Loss: {avg_loss:.4f}")

        # Map model IDs to fitnesses
        fitness_dict = {id(model): fitness for model, fitness in zip(population, fitnesses)}

        # Record fitness and diversity
        best_fitness = max(fitnesses)
        best_model_idx = fitnesses.index(best_fitness)
        best_model_in_generation = population[best_model_idx]

        fitness_history.append(best_fitness)
        diversity = population_diversity(population)
        diversity_history.append(diversity)

        # Update the best model overall if the best fitness in this generation is higher
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_model_overall = best_model_in_generation

        # Adaptive Mutation Rate
        mutation_rate = adaptive_mutation_rate(fitness_history, base_mutation_rate)
        print(f"Adaptive Mutation Rate: {mutation_rate:.4f}")

        # Update the speciation threshold dynamically
        threshold = initial_threshold - ((initial_threshold - final_threshold) * gen / (generations - 1))
        print(f"Speciation Threshold: {threshold:.4f}")

        # Speciate the population using the dynamic threshold
        species_list = speciate_population(population, threshold=threshold)

        # Adjust fitnesses using fitness sharing
        adjusted_fitness_dict = {}
        for species in species_list:
            num_individuals = len(species)
            for model in species:
                adjusted_fitness = fitness_dict[id(model)] / num_individuals
                adjusted_fitness_dict[id(model)] = adjusted_fitness

        # Calculate average fitness for each species
        species_avg_fitness = []
        for species in species_list:
            species_fitnesses = [fitness_dict[id(model)] for model in species]
            avg_fitness = sum(species_fitnesses) / len(species_fitnesses)
            species_avg_fitness.append(avg_fitness)

        # Determine fitness threshold for extinction (e.g., bottom 20%)
        fitness_threshold = np.percentile(species_avg_fitness, 20)

        # Identify surviving and extinct species
        surviving_species = []
        extinct_species_count = 0
        for i, species in enumerate(species_list):
            if species_avg_fitness[i] >= fitness_threshold:
                surviving_species.append(species)
            else:
                extinct_species_count += 1
                print(f"Species {i + 1} has gone extinct.")

        # Introduce new species
        new_species = []
        for _ in range(extinct_species_count):
            # Create a new random model
            new_model = initialize_population(1)[0]
            # Add the new model to the population and fitness dictionaries
            population.append(new_model)
            fitness_dict[id(new_model)] = 0.0
            adjusted_fitness_dict[id(new_model)] = 0.0
            new_species.append([new_model])
            print("A new species has been introduced.")

        # Combine surviving species and new species
        species_list = surviving_species + new_species

        # Proceed with selection and reproduction using adjusted fitnesses
        new_population = []
        for species in species_list:
            # Get adjusted fitnesses for the current species
            species_adjusted_fitnesses = [adjusted_fitness_dict[id(model)] for model in species]

            # Selection within species
            num_parents = max(1, len(species) // 2)
            parents = select_parents(species, species_adjusted_fitnesses, num_parents)

            # Generate offspring within the species
            offspring = []
            while len(offspring) < len(species) - num_parents:
                if len(parents) > 1:
                    parent1, parent2 = random.sample(parents, 2)
                else:
                    parent1 = parent2 = parents[0]
                child = crossover_models(parent1, parent2)
                child = mutate_model(child, mutation_rate)
                offspring.append(child)

            # Add offspring to population and initialize their fitness
            for child in offspring:
                population.append(child)
                fitness_dict[id(child)] = 0.0
                adjusted_fitness_dict[id(child)] = 0.0

            # Combine parents and offspring for this species
            new_population.extend(parents + offspring)

        # Update the population to the new population
        population = new_population

    # Plot Fitness and Diversity Over Generations
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(generations), fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness over Generations')

    plt.subplot(1, 2, 2)
    plt.plot(range(generations), diversity_history)
    plt.xlabel('Generation')
    plt.ylabel('Population Diversity')
    plt.title('Diversity over Generations')

    plt.show()

    return best_model_overall, best_fitness_overall