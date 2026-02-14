"""
Neural Network with ReLU + Softmax Activation
==========================================
Implementation of a feedforward neural network from scratch using NumPy.
Uses ReLU activation in hidden layers and softmax in the output layer with cross entropy loss.

Architecture: [784, 64, 64, 10]
- Input: 784 features (28x28 MNIST images flattened)
- Hidden layers: 2 layers of 64 neurons each
- Output: 10 neurons (one per digit class 0-9)
"""

import pickle
import numpy as np
import random
import pandas as pd
import os
import time
from typing import List, Tuple

class Network:
    def __init__(self, sizes: List[int], name: str) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Initialize biases randomly for each layer (except input)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # Initialize weights with He initialization: scale by sqrt(2/n_in)
        # This prevents vanishing/exploding gradients with ReLU activations
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.model_name = name
        self.rows = []   # Store metrics for each epoch

    def feedforward(self, a: np.ndarray) -> np.ndarray: # Compute network output for input a.
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            if layer_index == len(self.weights)-1:
                a = z # Output layer uses  softmax, so nothing is applied here
            else:
                a = relu(z) # Hidden layers use standard ReLU
        return a
    
    def SGD(self, 
        training_data: list, 
        mini_batch_size: int, 
        eta: float, 
        epochs: int, 
        test_data: list = None) -> None:
        # Train the network using Stochastic Gradient Descent
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            epoch_start = time.time() 
            random.shuffle(training_data) # Shuffle training data for each epoch

            # Split into mini-batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            # Update weights using each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Evaluate on test set if provided    
            if test_data: 
                val_acc = self.accuracy(test_data)
                val_loss = self.cross_entropy_loss(test_data)
                epoch_time = time.time() - epoch_start

                # Store metrics
                line = {"epoch": epoch,
                        "acc": val_acc,
                        "loss": val_loss,
                        "time": epoch_time,
                        "model": self.model_name
                }
                self.rows.append(line)
                print(f"Epoch {epoch}: accuracy = {val_acc*100:.2f}%  loss = {val_loss:.4f}  time = {epoch_time: .3f} sec")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, # Update network weights and biases using one mini-batch.
                         mini_batch: List[Tuple[np.ndarray, np.ndarray]], 
                         eta: float) -> None: 
        vector_b = [np.zeros(b.shape) for b in self.biases]
        vector_w = [np.zeros(w.shape) for w in self.weights]

        # Accumulate gradients from each example in the batch 
        for x,y in mini_batch:
            gradient_b, gradient_w = self.backprop(x,y)
            vector_b = [vb + gb for vb, gb in zip(vector_b, gradient_b)]
            vector_w = [vw + gw for vw, gw in zip(vector_w, gradient_w)]

        # Update weights and biases using averaged gradients
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, vector_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, vector_b)]
    
    def backprop(self, 
                x: np.ndarray, 
                y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        # Gradient containers 
        vector_b = [np.zeros(b.shape) for b in self.biases]
        vector_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass 
        activation = x
        activations = [x]
        vector_z = []
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activation)+b
            vector_z.append(z)

             # Apply activation function
            if layer_index == len(self.weights)-1:
                activation = z
            else:
                activation = relu(z)
            activations.append(activation)

        # Backward pass
        # Output layer gradient, where softmax applied
        p = softmax(activations[-1])
        delta = p - y
        vector_b[-1] = delta
        vector_w[-1] = np.dot(delta, activations[-2].transpose())

        # Hidden layers: backpropagate the error
        for j in range(2, self.num_layers):
            z = vector_z[-j]
            delta = np.dot(self.weights[-j+1].transpose(), delta)*relu_prime(z)
            vector_b[-j] = delta
            vector_w[-j] = np.dot(delta, activations[-j-1].transpose())
        return vector_b, vector_w
    
    def accuracy(self, data: List[Tuple[np.ndarray, int | np.ndarray]]) -> float: # Calculate classification accuracy on a dataset.
        correct = 0
        for x, y in data:
            prediction = int(np.argmax(self.feedforward(x))) # Highest activation
            correct += int(prediction == y)
        return correct / len(data)
    
    def cross_entropy_loss(self, data: List[Tuple[np.ndarray, int | np.ndarray]]) -> float: # Compute cross entropy loss with respect to output logits.
        total = 0.0
        for x,y in data:
            logits = self.feedforward(x)
            p = softmax(logits)
            y_vec = y if y.shape == p.shape else vectorized_results(y)
            total += -float(np.sum(y_vec*np.log(p+1e-12)))
        return total / len(data)

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================ 

def relu(z: np.ndarray) -> np.ndarray: # ReLU (Rectified Linear Unit) activation function.
    return np.maximum(0.0, z)

def relu_prime(z: np.ndarray) -> np.ndarray: # Derivative of ReLU activation.
    return (z>0).astype(float)

def softmax(z: np.ndarray) -> np.ndarray: # Apply softmax to output activations
    z = z - np.max(z)
    expz = np.exp(z)
    return expz/np.sum(expz)

def load_data()-> Tuple[tuple, tuple, tuple]: # Pickle MNIST file loader
    with open("mnist.pkl", "rb") as f: 
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1" )
    return training_data, validation_data, test_data

def data_wraper()-> Tuple[list, list, list]: # MNIST dataset wraper 
    a, b, c = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in a[0]]
    training_results = [vectorized_results(y) for y in a[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in b[0]]
    validation_data = list(zip(validation_inputs, b[1]))
    test_inputs = [np.reshape(x,(784,1)) for x in c[0]]  
    test_data = list(zip(test_inputs, c[1]))
    return training_data, validation_data, test_data

def vectorized_results(j: int) -> np.ndarray: # Convert a digit label to a one-hot encoded vector.
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def export_metrics(rows: List[dict]) -> None: # Save training metrics to CSV file.
    df = pd.DataFrame(rows)
    out_csv = "nminst_results_softmax.csv"

    # Add header only if file is new or empty
    header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
    df.to_csv(out_csv,mode="a", header=header, index=False)

#============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

# Load and prepare data
if  __name__ == "__main__":
    # Load and prepare data
    training_data, validation_data, test_data = data_wraper()

    # Run 10 independent trials for statistical reliability
    for i in range(10):
        print(f"\n=== Trial {i + 1} ===")

        # Create network with architecture [784, 64, 64, 10]
        net = Network([784,64, 64,10], "softmax_relu")

        # Train using SGD:
        # - Mini-batch size: 10
        # - Learning rate: 0.02
        # - Epochs: 11 (0-10)
        # - Evaluate on test set after each epoch
        net.SGD(training_data, 10, 0.02, 11, test_data)

        # Save results
        export_metrics(net.rows)
    print("Training complete!")
