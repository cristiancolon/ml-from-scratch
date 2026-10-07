import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
            self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X, self.weights) + self.biases
    def backward(self, dvals):
        self.dweights = np.dot(self.inputs.T, dvals)
        self.dbiases = np.sum(dvals, axis=0, keepdims=True)
        self.dinputs = np.dot(dvals, self.weights.T)
