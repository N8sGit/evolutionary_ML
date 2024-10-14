from .population import initialize_population, population_diversity
from training import train_model
from .fitness import evaluate_fitness
from .selection import select_parents, crossover_models
from .mutation import mutate_model, adaptive_mutation_rate
import torch
from data import train_loader, val_loader
import matplotlib.pyplot as plt
import random

def evolutionary_algorithm(pop_size=10, generations=5, base_mutation_rate=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    population = initialize_population(pop_size)
    
    fitness_history = []
    diversity_history = []

    best_fitness_overall = -float('inf')  # Initialize to a very low value
    best_model_overall = None  # To store the best model across all generations

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
            print(f"Model {i + 1} - Fitness: {fitness:.4f}, Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
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

        # Selection
        num_parents = pop_size // 2
        parents = select_parents(population, fitnesses, num_parents)
        
        # Generate offspring through crossover and mutation using list comprehension
        offspring = [mutate_model(crossover_models(*random.sample(parents, 2) if len(parents) > 1 else (parents[0], parents[0])), mutation_rate) for _ in range(pop_size - num_parents)]
        
        # Create new population
        population = parents + offspring

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