import numpy as np
from activation import Softmax

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)
    def regularization_loss(self, layer):
        w_l1 = layer.weight_l1reg * np.sum(np.abs(layer.weights))
        b_l1 = layer.bias_l1reg * np.sum(np.abs(layer.biases)) if layer.biases is not None else 0
        w_l2 = layer.weight_l2reg * np.sum(layer.weights ** 2)
        b_l2 = layer.bias_l2reg * np.sum(layer.biases ** 2) if layer.biases is not None else 0
        return w_l1 + b_l1 + w_l2 + b_l2

class CategoricalCrossEntropy(Loss):
    def forward(self, output, y):
        probs = np.clip(output, 1e-7, 1 - 1e-7)
        if y.shape == output.shape:
            return -np.sum(y * np.log(probs), axis=-1)
        correct = np.take_along_axis(probs, y[..., None], axis=-1).squeeze(-1)
        return -np.log(correct)
    def backward(self, dvals, y):
        if y.shape != dvals.shape:
            y = np.eye(dvals.shape[-1])[y]
        probs = np.clip(dvals, 1e-7, 1 - 1e-7)
        n = np.prod(dvals.shape[:-1])
        self.dinputs = -y / probs / n

class BinaryCrossEntropy(Loss):
    def forward(self, output, y):
        output_clipped = np.clip(output, 1e-7, 1-1e-7)
        sample_losses = -(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped))
        return np.mean(sample_losses, axis=-1)
    def backward(self, dvals, y):
        probs = np.clip(dvals, 1e-7, 1 - 1e-7)
        self.dinputs = (-y / probs + (1 - y) / (1 - probs)) / dvals.size

class MeanSquaredError(Loss):
    def forward(self, output, y):
        return np.mean((y - output) ** 2, axis=-1)
    def backward(self, dvals, y):
        self.dinputs = 2 * (dvals - y) / dvals.size

class MeanAbsoluteError(Loss):
    def forward(self, output, y):
        return np.mean(np.abs(y - output), axis=-1)
    def backward(self, dvals, y):
        self.dinputs = np.sign(dvals - y) / dvals.size