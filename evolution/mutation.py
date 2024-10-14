import torch
import torch.nn as nn
import random
import copy

def mutate_model(model, mutation_rate=0.1, weight_mutation_rate=0.05):
    """
    Mutates the given model by applying structural, hyperparameter, architecture, and weight mutations.

    Parameters:
    - model (nn.Module): The model to be mutated.
    - mutation_rate (float): The probability of applying any of the mutation strategies (layers, hyperparameters, architecture).
    - weight_mutation_rate (float): The probability of mutating individual weights in the model.

    Returns:
    - mutated_model (nn.Module): The mutated copy of the original model.
    """
    mutated_model = copy.deepcopy(model)
    
    # Mutate layers (structure)
    if random.random() < mutation_rate:
        mutated_model = mutate_layers(mutated_model)
    
    # Mutate hyperparameters
    if random.random() < mutation_rate:
        mutated_model = mutate_hyperparameters(mutated_model)
    
    # Mutate architecture-specific parameters
    if random.random() < mutation_rate:
        mutated_model = mutate_architecture(mutated_model)
    
    # Mutate weights
    with torch.no_grad():
        for layer in mutated_model.layer_list:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                weight_mask = torch.rand(layer.weight.size()) < weight_mutation_rate
                bias_mask = torch.rand(layer.bias.size()) < weight_mutation_rate
                layer.weight.add_(torch.randn(layer.weight.size()) * 0.1 * weight_mask)
                layer.bias.add_(torch.randn(layer.bias.size()) * 0.1 * bias_mask)
    
    return mutated_model

def mutate_layers(model):
    """
    Mutates the layers of the model by either adding or removing a hidden layer.
    If a layer is removed, adjacent layers are adjusted accordingly.

    Parameters:
    - model (nn.Module): The model whose layers will be mutated.

    Returns:
    - model (nn.Module): The mutated model.
    """
    original_layer_specs = copy.deepcopy(model.layer_specs)
    action = random.choice(['add', 'remove'])
    
    if action == 'remove' and len(model.layer_specs) > 4:
        # Identify removable layers (hidden Linear layers)
        removable_indices = [i for i, (layer_type, _) in enumerate(model.layer_specs)
                            if layer_type == 'Linear' and 0 < i < len(model.layer_specs) - 1]
        if removable_indices:
            remove_idx = random.choice(removable_indices)
            # Remove the Linear layer and associated activation and dropout layers
            del model.layer_specs[remove_idx:remove_idx + 3]
            # Adjust in_features of the next layer
            adjust_layer_dimensions(model, remove_idx - 1)
    else:
        # Add a hidden layer
        insert_idx = random.randint(1, len(model.layer_specs) - 2)
        prev_out_features = model.layer_specs[insert_idx - 1][1]['out_features']
        new_hidden_size = random.choice([64, 128, 256])
        new_layer_specs = [
            ('Linear', {'in_features': prev_out_features, 'out_features': new_hidden_size}),
            ('ReLU', {}),
            ('Dropout', {'p': random.uniform(0.1, 0.5)}),
        ]
        model.layer_specs[insert_idx:insert_idx] = new_layer_specs
        # Adjust in_features of the next layer
        adjust_layer_dimensions(model, insert_idx + 2)
    
    # Rebuild and verify the model
    if not rebuild_and_verify(model, original_layer_specs):
        model.layer_specs = original_layer_specs
        model.build_layers()
    return model

def adjust_layer_dimensions(model, idx):
    """
    Adjusts the input and output features of the next Linear layer after a mutation.
    
    Parameters:
    - model (nn.Module): The model whose layer dimensions are being adjusted.
    - idx (int): The index of the layer whose output features will be propagated to the next layer.
    """
    for i in range(idx + 1, len(model.layer_specs)):
        if model.layer_specs[i][0] == 'Linear':
            prev_out_features = model.layer_specs[idx][1]['out_features']
            model.layer_specs[i][1]['in_features'] = prev_out_features
            break

def rebuild_and_verify(model, original_layer_specs):
    """
    Rebuilds the model layers and verifies that the model dimensions remain consistent after mutations.

    Parameters:
    - model (nn.Module): The model being rebuilt.
    - original_layer_specs (list): The original layer specifications before the mutation.

    Returns:
    - (bool): True if the rebuild and verification are successful, otherwise False.
    """
    try:
        model.build_layers()
        model.verify_dimensions()
        return True
    except Exception as e:
        print(f"Error during model rebuild: {e}")
        return False

def ensure_layer_compatibility(model):
    """
    Ensures that all subsequent layers have compatible input and output features after a mutation.
    
    Parameters:
    - model (nn.Module): The model whose layers are being checked for compatibility.

    Returns:
    - model (nn.Module): The model with adjusted layer dimensions to ensure compatibility.
    """
    for idx in range(1, len(model.layer_specs)):
        if model.layer_specs[idx][0] == 'Linear' and model.layer_specs[idx - 1][0] == 'Linear':
            prev_out_features = model.layer_specs[idx - 1][1]['out_features']
            if model.layer_specs[idx][1]['in_features'] != prev_out_features:
                model.layer_specs[idx][1]['in_features'] = prev_out_features
    return model

def mutate_hyperparameters(model, mutation_rate=0.1):
    """
    Mutates the hyperparameters of the model, including dropout rates and activation functions.
    
    Parameters:
    - model (nn.Module): The model whose hyperparameters will be mutated.
    - mutation_rate (float): The probability of mutating any of the hyperparameters.

    Returns:
    - model (nn.Module): The model with mutated hyperparameters.
    """
    if random.random() < mutation_rate:
        # Mutate dropout rates in the model
        for idx, (layer_type, params) in enumerate(model.layer_specs):
            if layer_type == 'Dropout':
                new_dropout = random.uniform(0.1, 0.5)
                model.layer_specs[idx][1]['p'] = new_dropout
        
        # Mutate activation functions
        activation_choices = ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']
        for idx, (layer_type, _) in enumerate(model.layer_specs):
            if layer_type in ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU']:
                new_activation = random.choice(activation_choices)
                model.layer_specs[idx] = (new_activation, {})
        
        # Mutate optimizer hyperparameters if stored
        if hasattr(model, 'optimizer_params'):
            for param, value in model.optimizer_params.items():
                # Mutate learning rate, momentum, etc.
                if isinstance(value, float):
                    model.optimizer_params[param] = value * random.uniform(0.8, 1.2)
    return model

def mutate_architecture(model, mutation_rate=0.1):
    """
    Mutates the architecture of the model by adding or removing convolutional layers.

    Parameters:
    - model (nn.Module): The model whose architecture will be mutated.
    - mutation_rate (float): The probability of mutating the architecture.

    Returns:
    - model (nn.Module): The mutated model.
    """
    if hasattr(model, 'num_conv_layers') and random.random() < mutation_rate:
        # Decide to add or remove a convolutional layer
        action = random.choice(['add', 'remove'])
        if action == 'add':
            model.num_conv_layers += 1
            new_filter_size = random.choice([3, 5, 7])
            model.conv_layer_specs.append({'out_channels': random.choice([16, 32, 64]),
                                        'kernel_size': new_filter_size})
        elif action == 'remove' and model.num_conv_layers > 1:
            model.num_conv_layers -= 1
            model.conv_layer_specs.pop()
        
        # Re-initialize model architecture
        old_state_dict = model.state_dict()
        model.__init__(num_conv_layers=model.num_conv_layers, conv_layer_specs=model.conv_layer_specs)
        try:
            model.load_state_dict(old_state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Partial state dict load due to architecture change: {e}")
    return model

def adaptive_mutation_rate(fitness_history, base_rate=0.1, max_rate=0.5, min_rate=0.01):
    """
    Adjusts the mutation rate dynamically based on the improvement in fitness between generations.

    Parameters:
    - fitness_history (list): The list of fitness values over generations.
    - base_rate (float): The base mutation rate.
    - max_rate (float): The maximum allowed mutation rate.
    - min_rate (float): The minimum allowed mutation rate.

    Returns:
    - (float): The adjusted mutation rate.
    """
    if len(fitness_history) < 2:
        return base_rate
    improvement = fitness_history[-1] - fitness_history[-2]
    if improvement < 0:
        # Decrease in fitness, increase mutation rate
        new_rate = min(max_rate, base_rate * 1.5)
    elif improvement < 0.01:
        # Minor improvement, slightly increase mutation rate
        new_rate = min(max_rate, base_rate * 1.1)
    else:
        # Good improvement, decrease mutation rate
        new_rate = max(min_rate, base_rate * 0.9)
    return new_rate