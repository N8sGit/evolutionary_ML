import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from data import val_loader

def compare_models(model1, model2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1.to(device)
    model2.to(device)
    
    metrics_model1 = evaluate_model_performance(model1, val_loader, device)
    metrics_model2 = evaluate_model_performance(model2, val_loader, device)
    
    print("\n### Model 1 (Evolutionary Best Model) Performance ###")
    print(f"Accuracy: {metrics_model1['accuracy']:.4f}, Loss: {metrics_model1['loss']:.4f}")
    print("Confusion Matrix:")
    print(metrics_model1['confusion_matrix'])
    print("Classification Report:")
    print(metrics_model1['classification_report'])

    print("\n### Model 2 (Standard Trained Model) Performance ###")
    print(f"Accuracy: {metrics_model2['accuracy']:.4f}, Loss: {metrics_model2['loss']:.4f}")
    print("Confusion Matrix:")
    print(metrics_model2['confusion_matrix'])
    print("Classification Report:")
    print(metrics_model2['classification_report'])
    
def evaluate_model_performance(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    all_preds = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss_total += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = loss_total / len(data_loader)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }