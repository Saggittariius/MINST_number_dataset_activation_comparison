"""
Neural Network with PyTorch
============================
Implementation of a feedforward neural network using PyTorch framework.
Uses ReLU activation in hidden layers and CrossEntropyLoss for training.

This serves as a comparison to the from-scratch NumPy implementations,
demonstrating PyTorch's automatic differentiation and GPU acceleration.

Architecture: [784, 64, 64, 10]
- Input: 784 features (28x28 MNIST images flattened)
- Hidden layers: 2 layers of 64 neurons each
- Output: 10 neurons (one per digit class 0-9)

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import time
import pandas as pd
import os
from typing import Tuple

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class Network(nn.Module): # Feedforward neural network using PyTorch's nn.Module.
    
    def __init__(self) -> None:
        # Define layers
        super(Network, self).__init__()
        self.fc1 = nn.Linear(784, 64) # First hidden layer: 784 → 64
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer: 64 → 64
        self.fc3 = nn.Linear(64, 10) # Output layer: 64 → 10
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the forward pass through the network.
        x = F.relu(self.fc1(x)) # Hidden layer 1 + ReLU
        x = F.relu(self.fc2(x)) # Hidden layer 2 + ReLU
        x = self.fc3(x)  # Output layer (linear, no activation)
        return x

def load_data()-> Tuple[tuple, tuple, tuple]: # Load MNIST dataset from pickle file.
    with open("mnist.pkl", "rb") as f: 
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1" )
    return training_data, validation_data, test_data

def prepare_data(batch_size: int = 10) -> Tuple[DataLoader, DataLoader]: # Prepare MNIST data for PyTorch training.
    train, val, test = load_data()

    # Convert NumPy arrays to PyTorch tensors
    train_x = torch.FloatTensor(train[0]) # Training images (50000, 784)
    train_y = torch.LongTensor(train[1]) # Training labels (50000,) - class indices
    

    test_x = torch.FloatTensor(test[0]) # Test images (10000, 784)
    test_y = torch.LongTensor(test[1]) # Test labels (10000,) - class indices

    # Create TensorDatasets (pairs images with labels)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    # Create DataLoaders (handles batching and shuffling)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model: nn.Module, 
                   train_loader: DataLoader, 
                   criterion: nn.Module, 
                   optimizer: optim.Optimizer, 
                   device: torch.device) -> None: # Train the model for one epoch.
    model.train() # Set model to training mode (enables dropout, etc.)

    for batch_x, batch_y in train_loader:

        # Move data to GPU if available
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass: compute predictions
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass: compute gradients and update weights
        optimizer.zero_grad() # Clear old gradients
        loss.backward() # Compute gradients
        optimizer.step() # Update weights using gradients

def evaluate(model: nn.Module, 
            test_loader: DataLoader, 
            criterion: nn.Module, 
            device: torch.device) -> Tuple[float, float]:
    # Evaluate model performance on test set.
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad(): # Disable gradient computation
        for batch_x, batch_y in test_loader:
            
            # Move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # Get predictions (class with highest score)
            _, predicted = torch.max(outputs, 1) # Returns (values, indices)
            total += batch_y.size(0) # Count samples
            correct += (predicted == batch_y).sum().item()
            
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader) # Average over batches
    return accuracy, avg_loss

def export_metrics(rows: list) -> None:
    # Save training metrics to CSV file.
    df = pd.DataFrame(rows)
    out_csv = "nminst_results_pytorch.csv"

    # Add header only if file is new or empty
    header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
    df.to_csv(out_csv, mode = "a", header=header, index=False)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":

    # Run 10 independent trials for statistical reliability
    for trial in range(10):
        print(f"\n=== Trial {trial + 1} ===")

        # Create new model for each trial (fresh random initialization)
        model = Network().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.02)

        # Define loss function and optimizer
        train_loader, test_loader = prepare_data(batch_size=10)

        print("Starting training...")
        
        # Track metrics for this trial
        rows = []
        for epoch in range(11):
            epoch_start = time.time()
            # Train for one epoch
            train_one_epoch(model, train_loader, criterion, optimizer, device)

            # Evaluate on test set
            acc, loss = evaluate(model, test_loader, criterion, device)
            epoch_duration = time.time() - epoch_start

            # Store metrics
            row = {
                    "epoch": epoch,
                    "acc": acc,
                    "loss": loss,
                    "time": epoch_duration,
                    "model": "pytorch_relu"
                }
            rows.append(row)
            print(f"Epoch {epoch}: accuracy = {acc*100:.2f}%  loss = {loss:.4f}")

        # Save this trial's results
        export_metrics(rows)

    print(f"Training complete!")

            

    