import torch
import torch.nn as nn
import random
import copy

def mutate_model(model, mutation_rate=0.1):
    mutated_model = copy.deepcopy(model)
    if random.random() < mutation_rate:
        mutated_model = mutate_layers(mutated_model)
    # Mutate weights
    with torch.no_grad():
        for layer in mutated_model.layer_list:
            if isinstance(layer, nn.Linear):
                if random.random() < mutation_rate:
                    layer.weight.add_(torch.randn(layer.weight.size()) * 0.1)
                    layer.bias.add_(torch.randn(layer.bias.size()) * 0.1)
    return mutated_model

def mutate_layers(model):
    # Decide whether to add or remove a layer
    if random.random() < 0.5 and len(model.layer_specs) > 4:
        # Remove a hidden layer
        linear_indices = [i for i, (layer_type, _) in enumerate(model.layer_specs) if layer_type == 'Linear']
        # Exclude first and last Linear layers (input and output layers)
        hidden_linear_indices = linear_indices[1:-1]
        
        if hidden_linear_indices:
            remove_idx = random.choice(hidden_linear_indices)
            # Remove the Linear layer and its associated activation and dropout layers
            del model.layer_specs[remove_idx:remove_idx+3]
            
            # Adjust the in_features of the next Linear layer (if any)
            prev_linear_idx = remove_idx - 1
            next_linear_idx = remove_idx
            if next_linear_idx < len(model.layer_specs) and model.layer_specs[next_linear_idx][0] == 'Linear':
                # Ensure we're adjusting a valid Linear layer
                if 'out_features' in model.layer_specs[prev_linear_idx][1]:
                    prev_out_features = model.layer_specs[prev_linear_idx][1]['out_features']
                    model.layer_specs[next_linear_idx][1]['in_features'] = prev_out_features
            
            # Rebuild the model and verify dimensions
            model.build_layers()
            model.verify_dimensions()
    
    else:
        # Add a hidden layer
        linear_indices = [i for i, (layer_type, _) in enumerate(model.layer_specs) if layer_type == 'Linear']
        # Choose a position between existing Linear layers
        if len(linear_indices) > 2:
            insert_idx = random.choice(linear_indices[1:-1])  # Exclude input and output layers
            prev_out_features = model.layer_specs[insert_idx - 1][1]['out_features']
            new_hidden_size = random.choice([64, 128, 256])
            
            # Create new layer specs for the new hidden layer
            new_layer_specs = [
                ('Linear', {'in_features': prev_out_features, 'out_features': new_hidden_size}),
                ('ReLU', {}),
                ('Dropout', {'p': random.choice([0.3, 0.5, 0.7])}),
            ]
            
            # Adjust the in_features of the next Linear layer
            next_linear_idx = insert_idx
            if model.layer_specs[next_linear_idx][0] == 'Linear':
                model.layer_specs[next_linear_idx][1]['in_features'] = new_hidden_size
            
            # Insert the new layer specs at the chosen position
            model.layer_specs[insert_idx:insert_idx] = new_layer_specs
            
            # Rebuild the model and verify dimensions
            model.build_layers()
            model.verify_dimensions()
    
    return model

def mutate_hyperparameters(model, mutation_rate=0.1):
    if random.random() < mutation_rate:
        # Randomly choose a new dropout rate
        new_dropout_rate = random.choice([0.3, 0.5, 0.7])
        model.dropout = nn.Dropout(new_dropout_rate)
        model.dropout.p = new_dropout_rate  # Ensure the dropout rate is updated
    return model

def mutate_architecture(model, mutation_rate=0.1):
    # Only mutate architecture if the model supports it
    if hasattr(model, 'num_conv_layers') and random.random() < mutation_rate:
        # Randomly decide to add or remove a convolutional layer
        if random.random() < 0.5 and model.num_conv_layers > 1:
            model.num_conv_layers -= 1
        else:
            model.num_conv_layers += 1
        # Update the model architecture accordingly
        model.__init__(num_conv_layers=model.num_conv_layers, num_filters=model.num_filters)
    return model

def adaptive_mutation_rate(fitness_history, base_rate=0.1):
    # Adjust mutation rate based on fitness improvement
    if len(fitness_history) < 2:
        return base_rate
    improvement = fitness_history[-1] - fitness_history[-2]
    if improvement < 0.001:
        return min(1.0, base_rate * 1.5)  # Increase mutation rate
    else:
        return max(0.01, base_rate * 0.7)  # Decrease mutation rate