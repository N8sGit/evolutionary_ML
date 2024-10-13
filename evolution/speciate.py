import torch 

def speciate_population(population):
    # Group models into species based on architecture
    species_dict = {}
    for model in population:
        # Use architecture as key (e.g., hidden size and dropout rate)
        hidden_size = model.fc1.out_features
        dropout_rate = model.dropout.p
        arch_key = (hidden_size, dropout_rate)
        if arch_key not in species_dict:
            species_dict[arch_key] = []
        species_dict[arch_key].append(model)
    species = list(species_dict.values())
    return species

def is_similar(model1, model2, threshold=0.1):
    diff = 0
    params1 = model1.state_dict()
    params2 = model2.state_dict()
    for key in params1.keys():
        diff += torch.norm(params1[key] - params2[key])
    return diff.item() < threshold