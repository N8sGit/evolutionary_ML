from model import FlexibleNN
import random
import copy

def select_parents(population, fitnesses, num_parents, tournament_size=3):
    """
    Selects parents from the population using tournament selection.

    Parameters:
    - population (list): The list of models in the population.
    - fitnesses (list): The fitness scores corresponding to the population.
    - num_parents (int): The number of parents to select.
    - tournament_size (int): The number of candidates to compete in each tournament.

    Returns:
    - selected_parents (list): The selected parents from the population.
    """
    selected_parents = []
    population_fitness = list(zip(population, fitnesses))
    
    while len(selected_parents) < num_parents:
        # Adjust the tournament size if the remaining population is smaller than the tournament size
        if tournament_size > len(population_fitness):
            tournament_size = len(population_fitness)
        
        # Select random individuals from the population for the tournament
        tournament = random.sample(population_fitness, tournament_size)
        
        # Choose the individual with the highest fitness from the tournament
        winner = max(tournament, key=lambda x: x[1])[0]
        
        # Add the winner to the selected parents
        selected_parents.append(winner)
        
        # Remove the winner from the population to avoid selecting the same individual again
        population_fitness.remove((winner, max(tournament, key=lambda x: x[1])[1]))

    return selected_parents

def crossover_models(parent1, parent2):
    # Perform crossover on layer_specs
    min_length = min(len(parent1.layer_specs), len(parent2.layer_specs))
    if min_length < 2:
        crossover_point = 1
    else:
        crossover_point = random.randint(1, min_length - 1)
    # Combine layer_specs
    child_layer_specs = copy.deepcopy(parent1.layer_specs[:crossover_point]) + copy.deepcopy(parent2.layer_specs[crossover_point:])
    # Adjust dimensions
    for idx in range(1, len(child_layer_specs)):
        prev_layer_type, prev_params = child_layer_specs[idx - 1]
        curr_layer_type, curr_params = child_layer_specs[idx]
        if prev_layer_type == 'Linear' and curr_layer_type == 'Linear':
            prev_out_features = prev_params['out_features']
            curr_in_features = curr_params['in_features']
            if prev_out_features != curr_in_features:
                curr_params['in_features'] = prev_out_features
        elif prev_layer_type == 'Linear' and curr_layer_type != 'Linear':
            # For non-linear layers, no adjustment needed
            continue
        elif prev_layer_type != 'Linear' and curr_layer_type == 'Linear':
            # Find the last Linear layer before current
            for j in range(idx - 1, -1, -1):
                if child_layer_specs[j][0] == 'Linear':
                    prev_out_features = child_layer_specs[j][1]['out_features']
                    curr_params['in_features'] = prev_out_features
                    break
    # Ensure output layer has correct output size
    if child_layer_specs[-1][0] == 'Linear':
        child_layer_specs[-1][1]['out_features'] = 10  # Number of classes
    # Create child model
    child = FlexibleNN(layer_specs=child_layer_specs)
    # Verify dimensions
    child.verify_dimensions()
    return child