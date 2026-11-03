import numpy as np

class ReLU:
    def forward(self, X):
        self.inputs = X
        self.output = np.maximum(0, X)
    def backward(self, dvals):
        self.dinputs = dvals.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax:
    def forward(self, X):
        exp = np.exp(X - np.max(X, axis=-1, keepdims=True))
        self.output = exp / np.sum(exp, axis=-1, keepdims=True)
    def backward(self, dvals):
        self.dinputs = self.output * (dvals - np.sum(dvals * self.output, axis=-1, keepdims=True))

class Sigmoid:
    def forward(self, X): 
        self.output = 1 / (1 + np.exp(-X))
    def backward(self, dvals):
        self.dinputs = dvals * self.output * (1 - self.output)

class GELU:
    def forward(self, X):
        self.inputs = X
        self.output = 0.5 * X * (1 + np.tanh((2/np.pi)**0.5 * (X + 0.044715 * np.power(X, 3))))
    def backward(self, dvals):
        g = (2/np.pi)**0.5 * (self.inputs + 0.044715 * np.power(self.inputs, 3))
        g_prime = (2/np.pi)**0.5 * (1+0.134145 * np.power(self.inputs, 2))
        gelu_prime = 0.5 * (1 + np.tanh(g)) + 0.5 * self.inputs * (1 - np.power(np.tanh(g), 2)) * g_prime
        self.dinputs = dvals * gelu_prime