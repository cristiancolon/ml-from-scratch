import numpy as np
from data_generation import spiral_data
from layer import Dense
from activation import Softmax, ReLU
from loss import Softmax_CategoricalCrossEntropy, CategoricalCrossEntropy
from optimizer import SGD, MomentumSGD, AdaGrad, RMSProp, Adam
np.random.seed(0)


X, y = spiral_data(points=100, classes=3)

dense1 = Dense(2, 200)
activation1 = ReLU()
dense2 = Dense(200, 200)
activation2 = ReLU()
dense3 = Dense(200, 3)
activation_loss = Softmax_CategoricalCrossEntropy()
optim = Adam(0.1, 1e-3, 1e-7, 0.9, 0.999)

for i in range(100001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = activation_loss.forward(dense3.output, y)

    if i % 1000 == 0:
        print('epoch', i)
        print('loss:', loss)
        preds = np.argmax(activation_loss.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(preds == y)
        print('accuracy:', accuracy)

    activation_loss.backward(activation_loss.output, y)
    dense3.backward(activation_loss.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optim.update_params(dense3)
    optim.update_params(dense2)
    optim.update_params(dense1)
    optim.update_epochs()
