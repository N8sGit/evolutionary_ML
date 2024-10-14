import torch
import torch.nn as nn
from .population import evaluate_novelty
from sklearn.metrics import f1_score

def evaluate_fitness(model, device, val_loader, population=None, novelty_weight=0.1):
    model.eval()
    all_preds = []
    all_labels = []
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss_total += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Calculate F1-Score
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Compute average loss
    avg_loss = loss_total / len(val_loader)
    
    # Compute model complexity
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute novelty if population is provided
    if population is not None:
        novelty = evaluate_novelty(model, population)
    else:
        novelty = 0

    # Composite fitness: based on F1-score, model complexity, and novelty
    fitness = f1 / (num_params ** 0.5) + novelty_weight * novelty
    
    return fitness, f1, avg_loss