import torch
import torch.nn as nn
from .population import evaluate_novelty

def evaluate_fitness(model, device, val_loader, population=None, novelty_weight=0.1):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    loss_total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = correct / total
    avg_loss = loss_total / len(val_loader)
    # Compute model complexity
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Compute novelty if population is provided
    if population is not None:
        novelty = evaluate_novelty(model, population)
    else:
        novelty = 0
    # Composite fitness: higher accuracy, lower complexity, and higher novelty
    fitness = accuracy / (num_params ** 0.5) + novelty_weight * novelty
    return fitness, accuracy, avg_loss