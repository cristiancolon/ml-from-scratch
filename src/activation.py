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
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
    def backward(self, dvals):
        self.dinputs = np.empty_like(dvals)
        for idx, (single_out, single_dvals) in enumerate(zip(self.output, dvals)):
            single_out = single_out.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_out) - np.dot(single_out, single_out.T)
            self.dinputs[idx] = np.dot(jacobian_matrix, single_dvals)
