import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train(train_loader, model, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()              # Reset gradient
            outputs = model(inputs)            # Forward pass
            loss = criterion(outputs.view(-1, 9), labels.view(-1))  # Reshape outputs and labels correctly
            loss.backward()                    # Backward pass
            optimizer.step()                   # Update weights
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
