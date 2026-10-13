import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons, weight_l1reg, bias_l1reg, weight_l2reg, bias_l2reg):
            self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
            self.weight_l1reg = weight_l1reg
            self.bias_l1reg = bias_l1reg
            self.weight_l2reg = weight_l2reg
            self.bias_l2reg = bias_l2reg
    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X, self.weights) + self.biases
    def backward(self, dvals):
        self.dweights = np.dot(self.inputs.T, dvals)
        self.dbiases = np.sum(dvals, axis=0, keepdims=True)
        self.dinputs = np.dot(dvals, self.weights.T)

        d_l1 = np.ones_like(self.weights)
        d_l1[self.weights < 0] = -1
        self.dweights += self.weight_l1reg * d_l1
        d_l1 = np.ones_like(self.biases)
        d_l1[self.biases < 0] = -1
        self.dbiases += self.bias_l1reg * d_l1

        d_l2 = 2 * self.weights
        self.dweights += self.weight_l2reg * d_l2
        d_l2 = 2 * self.biases
        self.dbiases += self.bias_l2reg * d_l2

class Dropout:
    def __init__(self, dropout_keep):
        self.dropout_keep = dropout_keep
    def forward(self, X):
        self.inputs = X
        self.bernoullis = np.random.binomial(1, self.dropout_keep, size=self.inputs.shape)
        self.output = self.inputs * self.bernoullis / self.dropout_keep
    def backward(self, dvals):
        self.dinputs = dvals * self.bernoullis / self.dropout_keep
