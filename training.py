import torch
import torch.nn as nn
import torch.optim as optim
from model import FlexibleNN
from data import val_loader, train_loader, input_size
from evolution.fitness import evaluate_fitness
from evolution.population import create_random_architecture

import matplotlib.pyplot as plt


def evaluate_model(model, device, val_loader):
    model.eval()
    correct = 0
    total = 0
    epoch_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_accuracy = correct / total
    val_loss = epoch_loss / len(val_loader)
    
    return val_accuracy, val_loss

def plot_metrics(metric_history, metric_name):
    plt.figure()
    if 'evolutionary' in metric_history:
        plt.plot(metric_history['evolutionary'], label='Evolutionary Algorithm')
    if 'standard' in metric_history:
        plt.plot(metric_history['standard'], label='Standard Training')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Epochs')
    plt.legend()
    plt.show()

def train_model(model, device, train_loader, epochs=1, learning_rate=0.01, optimizer_type='SGD'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.to(device)
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    model.cpu()

def train_standard_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_list = create_random_architecture()
    model = FlexibleNN(layer_specs=layer_list, input_size=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    num_epochs = 5
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        train_accuracy = correct / total
        train_acc_history.append(train_accuracy)
        train_loss_history.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_accuracy, val_loss = evaluate_model(model, device, val_loader)
        val_acc_history.append(val_accuracy)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    # Return the trained model and its metrics
    return {
        'model': model,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'train_loss': train_loss_history,
        'val_loss': val_loss_history
    }