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
    # Save a copy of the original layer specs before mutation for reversion if needed
    original_layer_specs = copy.deepcopy(model.layer_specs)
    
    # Decide whether to add or remove a layer
    if random.random() < 0.5 and len(model.layer_specs) > 4:
        # Remove a hidden layer
        linear_indices = [i for i, (layer_type, _) in enumerate(model.layer_specs) if layer_type == 'Linear']
        hidden_linear_indices = linear_indices[1:-1]  # Exclude input and output layers
        
        if hidden_linear_indices:
            remove_idx = random.choice(hidden_linear_indices)
            # Remove the Linear layer and its associated activation and dropout layers
            del model.layer_specs[remove_idx:remove_idx + 3]
            
            # Adjust the in_features of the next Linear layer (if any)
            prev_linear_idx = remove_idx - 1
            next_linear_idx = remove_idx
            if next_linear_idx < len(model.layer_specs) and model.layer_specs[next_linear_idx][0] == 'Linear':
                # Safely access 'out_features' for the previous layer
                prev_out_features = model.layer_specs[prev_linear_idx][1].get('out_features', None)
                if prev_out_features is not None:
                    model.layer_specs[next_linear_idx][1]['in_features'] = prev_out_features

            # Rebuild the model and verify dimensions with error handling
            try:
                model.build_layers()
            except Exception as e:
                print(f"Error during layer construction: {e}")
                # Revert the model to the original state
                model.layer_specs = original_layer_specs
                model.build_layers()
                print("Reverted mutation due to error during layer construction.")
            
            # Handle dimension mismatch errors
            if not verify_and_handle(model, original_layer_specs):
                return model  # If reversion occurred, return the reverted model

    else:
        # Add a hidden layer
        linear_indices = [i for i, (layer_type, _) in enumerate(model.layer_specs) if layer_type == 'Linear']
        if len(linear_indices) > 2:
            insert_idx = random.choice(linear_indices[1:-1])  # Exclude input and output layers
            prev_out_features = model.layer_specs[insert_idx - 1][1].get('out_features', None)
            
            if prev_out_features is not None:
                new_hidden_size = random.choice([64, 128, 256])
                
                # Create new layer specs
                new_layer_specs = [
                    ('Linear', {'in_features': prev_out_features, 'out_features': new_hidden_size}),
                    ('ReLU', {}),
                    ('Dropout', {'p': random.choice([0.3, 0.5, 0.7])}),
                ]
                
                # Adjust the in_features of the next Linear layer
                next_linear_idx = insert_idx
                if model.layer_specs[next_linear_idx][0] == 'Linear':
                    model.layer_specs[next_linear_idx][1]['in_features'] = new_hidden_size

                # Insert the new layers
                model.layer_specs[insert_idx:insert_idx] = new_layer_specs
                
                # Rebuild the model and verify dimensions with error handling
                try:
                    model.build_layers()
                except Exception as e:
                    print(f"Error during layer construction: {e}")
                    # Revert the model to the original state
                    model.layer_specs = original_layer_specs
                    model.build_layers()
                    print("Reverted mutation due to error during layer construction.")

                # Handle dimension mismatch errors
                if not verify_and_handle(model, original_layer_specs):
                    return model  # If reversion occurred, return the reverted model
    
    return model

def verify_and_handle(model, original_layer_specs):
    """Verifies the dimensions of the model and reverts the mutation if an error is encountered."""
    try:
        model.verify_dimensions()
        return True
    except ValueError as e:
        print(f"Dimension mismatch detected: {e}")
        # Revert the model to the original state
        model.layer_specs = original_layer_specs
        model.build_layers()
        print("Reverted mutation due to dimension mismatch.")
        return False

def ensure_layer_compatibility(model):
    """Ensures that all subsequent layers have compatible in_features and out_features."""
    for idx in range(1, len(model.layer_specs)):
        if model.layer_specs[idx][0] == 'Linear' and model.layer_specs[idx - 1][0] == 'Linear':
            prev_out_features = model.layer_specs[idx - 1][1]['out_features']
            if model.layer_specs[idx][1]['in_features'] != prev_out_features:
                model.layer_specs[idx][1]['in_features'] = prev_out_features
    return model

def mutate_hyperparameters(model, mutation_rate=0.1):
    if random.random() < mutation_rate:
        # Randomly mutate hyperparameters like learning rate, dropout, and activation function
        new_dropout_rate = random.uniform(0.1, 0.7)
        model.dropout = nn.Dropout(new_dropout_rate)
        model.dropout.p = new_dropout_rate  # Ensure the dropout rate is updated

        if hasattr(model, 'learning_rate'):
            model.learning_rate *= random.uniform(0.8, 1.2)  # Mutate learning rate slightly

        if hasattr(model, 'activation_function'):
            model.activation_function = random.choice([nn.ReLU(), nn.Sigmoid(), nn.Tanh()])  # Change activation function
    return model

def mutate_architecture(model, mutation_rate=0.1):
    # Only mutate architecture if the model supports it
    if hasattr(model, 'num_conv_layers') and random.random() < mutation_rate:
        # Randomly decide to add or remove a convolutional layer
        if random.random() < 0.5 and model.num_conv_layers > 1:
            model.num_conv_layers -= 1
        else:
            model.num_conv_layers += 1
        # Update the model architecture while preserving weights where possible
        old_state_dict = model.state_dict()
        model.__init__(num_conv_layers=model.num_conv_layers, num_filters=model.num_filters)
        try:
            model.load_state_dict(old_state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to fully load the previous state dict: {e}")
    return model

def adaptive_mutation_rate(fitness_history, base_rate=0.1):
    # Adjust mutation rate based on fitness improvement
    if len(fitness_history) < 2:
        return base_rate
    improvement = fitness_history[-1] - fitness_history[-2]
    # Dynamically adjust mutation rate based on fitness trend
    if improvement < 0.001:
        return min(1.0, base_rate * (1.5 + random.uniform(0, 0.5)))  # Increase mutation rate with added randomness
    elif improvement < 0.01:
        return min(1.0, base_rate * 1.2)  # Slightly increase mutation rate
    else:
        return max(0.01, base_rate * (0.7 - random.uniform(0, 0.2)))  # Decrease mutation rate with added randomness