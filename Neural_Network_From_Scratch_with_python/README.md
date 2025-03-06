# Neural Network from Scratch in Python

## Overview
This project implements a fully connected neural network from scratch using python. It includes:
- Forward propagation
- Backpropagation
- Activation functions (ReLU, Sigmoid, Softmax)
- Categorical Crossentropy loss function
- Gradient Descent optimizer with learning rate decay and momentum

The neural network is tested on the `spiral_data` dataset from `nnfs.datasets`.

## Features
- One-hot encoding for labels
- He and Xavier weight initialization
- Multi-layer architecture with customizable activation functions
- Learning rate decay and momentum for optimization
- Accuracy and loss tracking
- Training visualization via Matplotlib

## Dependencies
Ensure you have the following installed:
```bash
pip install numpy matplotlib nnfs
```

## Usage
Run the script to train the neural network:
```bash
python nn.py
```
The model will train on the spiral dataset and generate a plot showing loss and accuracy over epochs.

## Architecture
The network consists of multiple dense layers:
- Input layer
- Hidden layers with ReLU activation
- Output layer with Softmax activation

## Explanation of Classes and Functions

### One-Hot Encoding
```python
def one_hot_encode(array:list):
```
This function converts class labels into a one-hot encoded format for categorical classification.

### Activation Functions
- **ReLU (Rectified Linear Unit):**
  ```python
  class ReLU:
  ```
  - Forward pass: Applies `max(0, x)` element-wise.
  - Backward pass: Computes the gradient, setting values ≤ 0 to 0.

- **Sigmoid:**
  ```python
  def sigmoid(x):
  ```
  - Computes the sigmoid function: `1 / (1 + exp(-x))`.
  - Used in binary classification (though not used in this model).

- **Softmax:**
  ```python
  class Softmax:
  ```
  - Converts logits into probability distributions by exponentiating and normalizing.
  
### Loss Function
- **Categorical Crossentropy:**
  ```python
  class Loss_CategoricalCrossentropy(Loss):
  ```
  - Measures the difference between predicted and actual class distributions.
  - Prevents log(0) errors by clipping values.
  
### Dense Layer
```python
class DenseLayer:
```
- Defines a fully connected layer.
- Supports He and Xavier weight initialization.
- Supports ReLU, Sigmoid, and Softmax activations.
- Performs forward propagation using matrix multiplication.
- Computes gradients during backpropagation.

### Neural Network Class
```python
class MultiClassification_Neural_Network:
```
- Builds a feedforward neural network with user-defined layers.
- Performs forward propagation through each layer.
- Computes loss using categorical crossentropy.
- Backpropagates errors through layers.

### Optimizer: Gradient Descent with Momentum and Decay
```python
class GradientDescent_Optimizer:
```
- Implements gradient descent with learning rate decay and momentum.
- Updates weights and biases after each training iteration.

### Training Process
```python
def main():
```
1. Generates a synthetic dataset using `spiral_data()`.
2. Initializes the neural network with specified layer sizes.
3. Runs training for a set number of epochs:
   - Performs forward propagation.
   - Computes loss and accuracy.
   - Backpropagates errors.
   - Updates parameters using gradient descent.
4. Tracks and plots accuracy and loss over time.

## Author
Developed as an educational project for deep learning fundamentals.