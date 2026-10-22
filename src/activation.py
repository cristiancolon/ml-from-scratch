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
