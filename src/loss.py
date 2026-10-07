import numpy as np
from activation import Softmax

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class CategoricalCrossEntropy(Loss):
    def forward(self, output, y):
        n = len(output)
        output_clipped = np.clip(output, 1e-7, 1-1e-7)
        if len(y.shape) == 1:
            correct_confidences = output_clipped[range(n), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(output_clipped * y, axis=1)
        return -np.log(correct_confidences)
    def backward(self, dvals, y):
        samples = len(dvals)
        labels = len(dvals[0])
        if len(y.shape) == 1: # If labels are sparse, turn them into one-hot vector
            y = np.eye(labels)[y]
        self.dinputs = -y / dvals
        self.dinputs = self.dinputs / samples

class Softmax_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
    def forward(self, X, y):
        self.activation.forward(X)
        self.output =  self.activation.output
        return self.loss.calculate(self.output, y)
    def backward(self, dvals, y):
        samples = len(dvals)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs = dvals.copy()
        self.dinputs[range(samples), y] -= 1
        self.dinputs = self.dinputs / samples