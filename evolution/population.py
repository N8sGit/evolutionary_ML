import random
import torch
import torch.nn as nn
from model import FlexibleNN
from data import input_size


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        # Create random architecture
        layer_specs = create_random_architecture()
        model = FlexibleNN(layer_specs=layer_specs, input_size=input_size)
        population.append(model)
    return population

def create_random_architecture():
    layer_specs = []
    input_size = 28*28
    output_size = 10
    hidden_sizes = [64, 128, 256]
    num_hidden_layers = random.randint(1, 3)
    # Input layer
    out_features = random.choice(hidden_sizes)
    layer_specs.append(('Linear', {'in_features': input_size, 'out_features': out_features}))
    layer_specs.append(('ReLU', {}))
    layer_specs.append(('Dropout', {'p': random.choice([0.3, 0.5, 0.7])}))
    # Hidden layers
    for _ in range(num_hidden_layers - 1):
        in_features = out_features
        out_features = random.choice(hidden_sizes)
        layer_specs.append(('Linear', {'in_features': in_features, 'out_features': out_features}))
        layer_specs.append(('ReLU', {}))
        layer_specs.append(('Dropout', {'p': random.choice([0.3, 0.5, 0.7])}))
    # Output layer
    layer_specs.append(('Linear', {'in_features': out_features, 'out_features': output_size}))
    return layer_specs

layer_diff_penalty = 10  # Define the penalty for different number of layers

def model_difference(model1, model2):
    diff = 0
    # Compare number of layers
    diff += abs(len(model1.layer_list) - len(model2.layer_list)) * 10  # Scale factor for significance
    # Compare layer parameters
    min_layers = min(len(model1.layer_list), len(model2.layer_list))
    for i in range(min_layers):
        layer1 = model1.layer_list[i]
        layer2 = model2.layer_list[i]
        if type(layer1) != type(layer2):
            diff += 5  # Penalty for different layer types
        else:
            if hasattr(layer1, 'weight') and hasattr(layer2, 'weight'):
                if layer1.weight.shape == layer2.weight.shape:
                    diff += torch.norm(layer1.weight - layer2.weight)
                else:
                    diff += torch.norm(layer1.weight) + torch.norm(layer2.weight)
    return diff.item()

def architecture_difference(model1, model2):
    diff = 0
    # Compare hidden sizes
    hidden_size1 = model1.fc1.out_features
    hidden_size2 = model2.fc1.out_features
    diff += abs(hidden_size1 - hidden_size2)
    # Compare dropout rates
    dropout_rate1 = model1.dropout.p
    dropout_rate2 = model2.dropout.p
    diff += abs(dropout_rate1 - dropout_rate2) * 100  # Scale factor for significance
    return diff

def evaluate_novelty(model, population):
    # Compute the average distance to other models
    distances = []
    for other_model in population:
        if other_model == model:
            continue
        diff = model_difference(model, other_model)
        distances.append(diff)
    # Novelty is the average distance to other models
    novelty = sum(distances) / len(distances) if distances else 0
    return novelty

def population_diversity(population):
    # Compute pairwise differences
    diversities = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            diff = model_difference(population[i], population[j])
            diversities.append(diff)
    avg_diversity = sum(diversities) / len(diversities) if diversities else 0
    return avg_diversity