from model import FlexibleNN
import random
import copy

def select_parents(population, fitnesses, num_parents):
    parents = [model for _, model in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
    return parents[:num_parents]


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