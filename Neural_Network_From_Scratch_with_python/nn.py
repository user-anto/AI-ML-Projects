import numpy as np
import matplotlib.pyplot as plt
import nnfs
import time

nnfs.init()

def one_hot_encode(array:list):
    one_hot_encoded_array = np.zeros((len(array), max(array)+1))
    for i, value in enumerate(array):
        one_hot_encoded_array[i, value] = 1
    return one_hot_encoded_array

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)

    def backward(self, dL_dz):
        self.dinp = dL_dz.copy()
        self.dinp[self.inputs <= 0] = 0

def Relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

class Softmax:
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.output = probs

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

def accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_true)
    return accuracy

class DenseLayer:
    def __init__(self,
                 n_inputs:int,
                 neurons:int,
                 activation_fn:str,
                 init:str):
        
        init = init.lower()
        
        def he_init(shape):
            fan_in, _ = shape
            std = np.sqrt(2 / fan_in)
            return np.random.randn(*shape) * std

        def xavier_init(shape):
            fan_in, fan_out = shape
            limit = np.sqrt(6 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, size=shape)
        
        if init == "he":
            self.weights = he_init((n_inputs, neurons))
        elif init == "xavier":
            self.weights = xavier_init((n_inputs, neurons))
        else:
            raise ValueError("Invalid parameter initialization method.")
        
        self.bias = np.zeros((1, neurons))
        self.activation_fn = activation_fn.lower()
        
        if self.activation_fn not in ["relu", "sigmoid", "softmax"]:
            raise NameError("Use activation functions only: ReLU, Sigmoid, Softmax\n")
        
    def forward(self, inputs):
        self.inputs = np.array(inputs)
        not_activation = np.dot(self.inputs, self.weights) + self.bias

        if self.activation_fn == "relu":
            relu = ReLU()
            relu.forward(not_activation)
            self.output = relu.output
        elif self.activation_fn == "sigmoid":
            self.output = sigmoid(not_activation)
        else:
            softmax = Softmax()
            softmax.forward(not_activation)
            self.output = softmax.output
        return self.output
    
    def backward(self, dL_dz):
        if self.activation_fn == "relu":
            dL_dz = np.where(self.output > 0, dL_dz, 0)
        elif self.activation_fn == "sigmoid":
            dL_dz *= self.output * (1 - self.output)
        
        self.dw = np.dot(self.inputs.T, dL_dz)
        self.db = np.sum(dL_dz, axis=0, keepdims=True)
        self.dinputs = np.dot(dL_dz, self.weights.T)

        return self.dinputs

class MultiClassification_Neural_Network:
    def __init__(self, n_layers:list, training_dataset:list):
        self.n_layers = n_layers
        self.data = training_dataset
        self.Layers = []

        for i in range(len(self.n_layers)):
            if i == 0:
                self.Layers.append(DenseLayer(n_inputs=len(self.data[0]), neurons=self.n_layers[0], activation_fn="ReLU", init="he"))
            elif i == len(self.n_layers) - 1:
                self.Layers.append(DenseLayer(n_inputs=self.n_layers[i-1], neurons=self.n_layers[i], activation_fn="Softmax", init="xavier"))
            else:
                self.Layers.append(DenseLayer(n_inputs=self.n_layers[i-1], neurons=self.n_layers[i], activation_fn="ReLU", init="he"))

    def forward_prop(self):
        layer_output = self.data
        for layer in self.Layers:
            layer_output = layer.forward(layer_output)
        self.output = layer_output

    def calculate_loss(self, y_true):
        loss_fn = Loss_CategoricalCrossentropy()
        self.loss = loss_fn.forward(y_pred=self.output, y_true=y_true)
        mean_loss = np.mean(self.loss)
        return mean_loss
    
    def backpropagate(self, y_true):
        dL_dz = self.output.copy()
        dL_dz[range(len(y_true)), y_true] -= 1
        dL_dz /= len(y_true)
        
        for layer in reversed(self.Layers):
            dL_dz = layer.backward(dL_dz)
        
        return dL_dz
    
class GradientDescent_Optimizer:
    def __init__(self, learning_rate=0.01, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def update_params(self, neuralnet):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)
        
        if self.current_learning_rate <= 0.015:
            self.current_learning_rate = 0.02

        if self.momentum:
            for layer in neuralnet.Layers:
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.bias)

                layer.weight_momentums = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dw
                layer.bias_momentums = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.db
                
                layer.weights += layer.weight_momentums
                layer.bias += layer.bias_momentums
        else:
            for layer in neuralnet.Layers:    
                layer.weights -= self.current_learning_rate * layer.dw
                layer.bias -= self.current_learning_rate * layer.db

        self.iterations += 1

def main():
    from nnfs.datasets import spiral_data
    data_length, n_classes = 100, 3
    X, y = spiral_data(samples=data_length, classes=n_classes)
    one_hot_encoded_y = one_hot_encode(y)

    NeuralNet = MultiClassification_Neural_Network([2, 128, 256, 128, 3], training_dataset=X)
    epochs = 10000
    lr = 0.1
    decay = 0.0005
    momentum = 0.9
    optimizer = GradientDescent_Optimizer(learning_rate=lr, decay=decay, momentum=momentum)

    start = time.time()

    epochs_history = np.arange(1, epochs+1)
    loss_history, accuracy_history = [], []

    for i in range(1, epochs+1):
        NeuralNet.forward_prop()

        loss = NeuralNet.calculate_loss(y_true=one_hot_encoded_y)

        NeuralNet.backpropagate(y_true=y)

        optimizer.update_params(NeuralNet)

        acc = accuracy(y_pred=NeuralNet.output, y_true=y)

        loss_history.append(loss)
        accuracy_history.append(acc)
        
        print(f"Epoch {i} || Loss: {loss:.3f}, Accuracy: {acc:.3f}, Learning rate: {optimizer.current_learning_rate:.3f}\n")

    end = time.time()

    print(f"Training time: {(end - start) // 60} mins. {((end - start) % 60):.4f} seconds.") # Takes 1.0 mins. 40.4751 seconds. (on i9 14900HX.)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_history, loss_history, label="Loss", marker='o', linestyle='-')
    plt.plot(epochs_history, accuracy_history, label="Accuracy", marker='s', linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Loss and Accuracy vs. Epochs")
    plt.legend()
    plt.grid(True)

    plt.savefig("loss_curve.png", dpi=400, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()