import torch 
import torch.nn as nn
import torch.nn.functional as F

def evaluate(model, test_loader ):
    model.eval()  # Met le modèle en mode évaluation
    correct = 0
    total = 0

    with torch.no_grad():  # Désactive le calcul des gradients pour l'évaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 2)  # Dimension 2: classe probabilités dans les sorties [batch_size, 16, 4]
            total += labels.numel()  # Total de chiffres évalués, labels.numel() devrait être batch_size*16
            correct += (predicted.view(-1) == labels.view(-1)).sum().item()  # Comparaison élément par élément

    accuracy = 100 * correct / total
    print(f"Accuracy du modèle: {accuracy}%")
    return accuracy
