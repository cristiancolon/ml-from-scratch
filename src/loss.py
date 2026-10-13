import numpy as np
from activation import Softmax

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)
    def regularization_loss(self, layer):
        w_l1 = layer.weight_l1reg * np.sum(np.abs(layer.weights))
        b_l1 = layer.bias_l1reg * np.sum(np.abs(layer.biases))
        w_l2 = layer.weight_l2reg * np.sum(layer.weights ** 2)
        b_l2 = layer.bias_l2reg * np.sum(layer.biases ** 2)
        return w_l1 + b_l1 + w_l2 + b_l2

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
        if len(y.shape) == 1:
            y = np.eye(labels)[y]
        self.dinputs = -y / dvals
        self.dinputs = self.dinputs / samples

class BinaryCrossEntropy(Loss):
    def forward(self, output, y):
        output_clipped = np.clip(output, 1e-7, 1-1e-7)
        sample_losses = -(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped))
        return np.mean(sample_losses, axis=-1)
    def backward(self, dvals, y):
        samples = len(dvals)
        outs = len(dvals[0])
        clipped_dvals = np.clip(dvals, 1e-7, 1 - 1e-7)
        self.dinputs = - (y / clipped_dvals - (1-y) / (1-clipped_dvals)) / outs
        self.dinputs = self.dinputs / samples

class MeanSquaredError(Loss):
    def forward(self, output, y):
        return np.mean((y - output) ** 2, axis=-1)
    def backward(self, dvals, y):
        samples = len(dvals)
        outs = len(dvals[0])
        self.dinputs = -2 * (y - dvals) / outs
        self.dinputs = self.dinputs / samples

class MeanAbsoluteError(Loss):
    def forward(self, output, y):
        return np.mean(np.abs(y - output), axis=-1)
    def backward(self, dvals, y):
        samples = len(dvals)
        outs = len(dvals[0])
        self.dinputs = -np.sign(y - dvals) / outs
        self.dinputs = self.dinputs / samples