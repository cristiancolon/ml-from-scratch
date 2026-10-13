import numpy as np
from data_generation import spiral_data
from layer import Dense, Dropout
from activation import Softmax, ReLU, Sigmoid
from loss import CategoricalCrossEntropy, BinaryCrossEntropy
from optimizer import SGD, MomentumSGD, AdaGrad, RMSProp, Adam

np.random.seed(0)

X, y = spiral_data(points=100, classes=2)
y = y.reshape(-1, 1)

dense1 = Dense(2, 200, 5e-4, 5e-4, 0, 0)
activation1 = ReLU()
dropout1 = Dropout(0.95)
dense2 = Dense(200, 200, 0, 0, 5e-4, 5e-4)
activation2 = ReLU()
dropout2 = Dropout(0.95)
dense3 = Dense(200, 1, 0, 0, 0, 0)
activation3 = Sigmoid()
loss_func = BinaryCrossEntropy()
optim = Adam(0.1, 5e-7, 1e-7, 0.9, 0.999)

for i in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output)
    dense3.forward(dropout2.output)
    activation3.forward(dense3.output)
    data_loss = loss_func.calculate(activation3.output, y)
    loss = data_loss + loss_func.regularization_loss(dense1) + loss_func.regularization_loss(dense2) + loss_func.regularization_loss(dense3)
    if i % 1000 == 0:
        print('epoch', i)
        print('loss:', loss)
        preds = (activation3.output > 0.5) * 1
        accuracy = np.mean(preds == y)
        print('accuracy:', accuracy)

    loss_func.backward(activation3.output, y)
    activation3.backward(loss_func.dinputs)
    dense3.backward(activation3.dinputs)
    dropout2.backward(dense3.dinputs)
    activation2.backward(dropout2.dinputs)
    dense2.backward(activation2.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optim.update_params(dense3)
    optim.update_params(dense2)
    optim.update_params(dense1)
    optim.update_epochs()
