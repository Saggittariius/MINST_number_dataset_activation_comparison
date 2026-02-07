# MNIST Activation Function Comparison

An experimental comparison of different activation functions for neural network training on the MNIST handwritten digit dataset.

## Models Compared

- **Sigmoid** - Traditional activation with standard initialization
- **ReLU** - Rectified Linear Unit with He initialization
- **Leaky ReLU** - ReLU variant that prevents dying neurons
- **Softmax + Cross-Entropy** - Modern approach with proper loss function

## Architecture

- Input: 784 pixels (28×28 images)
- Hidden layers: [64, 64]
- Output: 10 classes (digits 0-9)
- Training: SGD with mini-batches (batch size: 10, learning rate: 0.02)
- Runs: 10 independent trials per configuration

## Key Findings

- Leaky ReLU and Softmax achieved ~97% accuracy
- Standard ReLU showed high variance (28-78% accuracy)
- Sigmoid reached ~65-78% but suffered from slower convergence
- Proper loss function choice (cross-entropy vs MSE) matters significantly

## Repository Contents

- `neuralnet_*.py` - Neural network implementations for each activation function
- `minst_plots.R` - R script for generating comparison visualizations
- `*.csv` - Training results (accuracy and loss per epoch)
