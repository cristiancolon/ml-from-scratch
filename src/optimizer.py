import numpy as np
class SGD:
    def __init__(self, lr, decay):
        self.lr = lr
        self.decay = decay
        self.epochs = 0
    def update_params(self, layer):
        curr_lr = self.lr * 1 / (1 + self.epochs * self.decay)
        layer.weights -= curr_lr * layer.dweights
        layer.biases -= curr_lr * layer.dbiases
    def update_epochs(self):
        self.epochs += 1

class MomentumSGD(SGD):
    def __init__(self, lr, decay, momentum):
        super().__init__(lr, decay)
        self.momentum = momentum
    def update_params(self, layer):
        curr_lr = self.lr * 1 / (1 + self.epochs * self.decay)
        if not hasattr(layer, 'prev_updates'):
            layer.prev_updates = [np.zeros_like(layer.weights), np.zeros_like(layer.biases)]
        update_weights = self.momentum * layer.prev_updates[0] - curr_lr * layer.dweights
        update_biases = self.momentum * layer.prev_updates[1] - curr_lr * layer.dbiases
        self.prev_weight_update = update_weights
        self.prev_bias_update = update_biases
        layer.weights += update_weights
        layer.biases += update_biases

class AdaGrad(SGD):
    def __init__(self, lr, decay, epsilon):
        super().__init__(lr, decay)
        self.epsilon = epsilon
    def update_params(self, layer):
        curr_lr = self.lr * 1 / (1 + self.epochs * self.decay)
        if not hasattr(layer, 'caches'):
            layer.caches = [np.zeros_like(layer.weights), np.zeros_like(layer.biases)]
        layer.caches[0] += layer.dweights ** 2
        layer.caches[1] += layer.dbiases ** 2
        layer.weights -= curr_lr * layer.dweights / (np.sqrt(layer.caches[0]) + self.epsilon)
        layer.biases -= curr_lr * layer.dbiases / (np.sqrt(layer.caches[1]) + self.epsilon)

class RMSProp(SGD):
    def __init__(self, lr, decay, epsilon, rho):
        super().__init__(lr, decay)
        self.rho = rho
        self.epsilon = epsilon
    def update_params(self, layer):
        curr_lr = self.lr * 1 / (1 + self.epochs * self.decay)
        if not hasattr(layer, 'caches'):
            layer.caches = [np.zeros_like(layer.weights), np.zeros_like(layer.biases)]
        layer.caches[0] = self.rho * layer.caches[0] + (1 - self.rho) * layer.dweights ** 2
        layer.caches[1] = self.rho * layer.caches[1] + (1 - self.rho) * layer.dbiases ** 2
        layer.weights -= curr_lr * layer.dweights / (np.sqrt(layer.caches[0]) + self.epsilon)
        layer.biases -= curr_lr * layer.dbiases / (np.sqrt(layer.caches[1]) + self.epsilon)

class Adam(SGD):
    def __init__(self, lr, decay, epsilon, beta1, beta2):
        super().__init__(lr, decay)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    def update_params(self, layer):
        curr_lr = self.lr * 1 / (1 + self.epochs * self.decay)
        if not hasattr(layer, "caches"):
            layer.caches = [np.zeros_like(layer.weights), np.zeros_like(layer.biases)]
            layer.momentums = [np.zeros_like(layer.weights), np.zeros_like(layer.biases)]
        layer.momentums[0] = self.beta1 * layer.momentums[0] + (1 - self.beta1) * layer.dweights
        layer.momentums[1] = self.beta1 * layer.momentums[1] + (1 - self.beta1) * layer.dbiases

        layer.caches[0] = self.beta2 * layer.caches[0] + (1 - self.beta2) * layer.dweights ** 2
        layer.caches[1] = self.beta2 * layer.caches[1] + (1 - self.beta2) * layer.dbiases ** 2

        corrected_momentums = [layer.momentums[i] / (1 - self.beta1 ** (self.epochs+1)) for i in range(2)]
        corrected_caches = [layer.caches[i] / (1 - self.beta2 ** (self.epochs+1)) for i in range(2)]

        layer.weights -= curr_lr * corrected_momentums[0] / (np.sqrt(corrected_caches[0]) + self.epsilon)
        layer.biases -= curr_lr * corrected_momentums[1] / (np.sqrt(corrected_caches[1]) + self.epsilon)
        
