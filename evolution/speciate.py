import torch 

def get_architecture_key(model):
    # Generate a key that uniquely identifies the model's architecture
    # This function should capture all relevant architectural parameters
    layers = []
    for layer in model.layer_list:
        if isinstance(layer, torch.nn.Linear):
            layers.append(('Linear', layer.in_features, layer.out_features))
        elif isinstance(layer, torch.nn.Conv2d):
            layers.append(('Conv2d', layer.in_channels, layer.out_channels, layer.kernel_size))
        elif isinstance(layer, torch.nn.ReLU):
            layers.append('ReLU')
        elif isinstance(layer, torch.nn.Dropout):
            layers.append(('Dropout', layer.p))
        # Add other layer types as needed
    return tuple(layers)

def speciate_population(population, threshold=0.1):
    species = []
    # First, group models by architecture
    architecture_dict = {}
    for model in population:
        arch_key = get_architecture_key(model)
        if arch_key not in architecture_dict:
            architecture_dict[arch_key] = []
        architecture_dict[arch_key].append(model)
    
    # Then, within each architecture group, cluster models based on similarity
    for arch_models in architecture_dict.values():
        arch_species = []
        for model in arch_models:
            placed = False
            for s in arch_species:
                representative = s[0]
                if is_similar(model, representative, threshold):
                    s.append(model)
                    placed = True
                    break
            if not placed:
                arch_species.append([model])
        species.extend(arch_species)
    return species

def is_similar(model1, model2, threshold):
    params1 = model1.state_dict()
    params2 = model2.state_dict()
    diff = 0.0

    for key in params1.keys():
        # No need to check for key existence or size mismatch
        diff += torch.norm(params1[key] - params2[key])
    return diff < threshold