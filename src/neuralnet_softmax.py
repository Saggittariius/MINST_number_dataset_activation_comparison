import pickle
import numpy as np
import random
import pandas as pd
import os

class Network:
    def __init__(self, sizes, name):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(1/x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.model_name = name
        self.rows = []

    def feedforward(self, a):
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            if layer_index == len(self.weights)-1:
                 a = z
            else:
                a = relu(z)
        return a
    
    def SGD(self, training_data, mini_batch_size, eta, epochs, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs): 
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data: 
                val_acc = self.accuracy(test_data)
                val_loss = self.cross_entropy_loss(test_data)
                line = {"epoch": epoch,
                        "acc": val_acc,
                        "loss": val_loss,
                        "model": self.model_name
                }
                self.rows.append(line)
                print(f"Epoch {epoch}: accuracy = {val_acc*100:.2f}%  loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, eta):
        vector_b = [np.zeros(b.shape) for b in self.biases]
        vector_w = [np.zeros(w.shape) for w in self.weights] 
        for x,y in mini_batch:
            gradient_b, gradient_w = self.backprop(x,y)
            vector_b = [vb + gb for vb, gb in zip(vector_b, gradient_b)]
            vector_w = [vw + gw for vw, gw in zip(vector_w, gradient_w)]
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, vector_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, vector_b)]
    
    def backprop(self, x, y):
        # gradient containers 
        vector_b = [np.zeros(b.shape) for b in self.biases]
        vector_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass 
        activation = x
        activations = [x]
        vector_z = []
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activation)+b
            vector_z.append(z)
            if layer_index == len(self.weights)-1:
                activation = z
            else:
                activation = relu(z)
            activations.append(activation)

        p = softmax(activations[-1])
        delta = p - y
        vector_b[-1] = delta
        vector_w[-1] = np.dot(delta, activations[-2].transpose())

        for j in range(2, self.num_layers):
            z = vector_z[-j]
            delta = np.dot(self.weights[-j+1].transpose(), delta)*relu_prime(z)
            vector_b[-j] = delta
            vector_w[-j] = np.dot(delta, activations[-j-1].transpose())
        return vector_b, vector_w
    
    def accuracy(self, data):
        correct = 0
        for x, y in data:
            prediction = int(np.argmax(self.feedforward(x)))
            correct += int(prediction == y)
        return correct / len(data)
    
    def cross_entropy_loss(self, data):
        total = 0.0
        for x,y in data:
            logits = self.feedforward(x)
            p = softmax(logits)
            y_vec = y if y.shape == p.shape else vectorized_results(y)
            total += -float(np.sum(y_vec*np.log(p+1e-12)))
        return total / len(data)


def relu(z):
    return np.maximum(0.0, z)

def relu_prime(z):
    return (z>0).astype(float)

def softmax(z):
    z = z - np.max(z)
    expz = np.exp(z)
    return expz/np.sum(expz)

def load_data():
    with open("mnist.pkl", "rb") as f: 
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1" )
    return training_data, validation_data, test_data

def data_wraper():
    a, b, c = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in a[0]]
    training_results = [vectorized_results(y) for y in a[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in b[0]]
    validation_data = list(zip(validation_inputs, b[1]))
    test_inputs = [np.reshape(x,(784,1)) for x in c[0]]  
    test_data = list(zip(test_inputs, c[1]))
    return training_data, validation_data, test_data

def vectorized_results(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def export_metrics(rows):
    df = pd.DataFrame(rows)
    out_csv = "nminst_results_softmax.csv"
    header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
    df.to_csv(out_csv,mode="a", header=header, index=False)

training_data, validation_data, test_data = data_wraper()
for i in range(10):
    net = Network([784,64, 64,10], "softmax_relu")
    net.SGD(training_data, 10, 0.02, 11, test_data)
    export_metrics(net.rows)
