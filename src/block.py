import numpy
from layer import Linear, MultiHeadAttention, RMSNorm, Dropout
from activation import GELU

class Transformer:
    def __init__(self, d_model, emb_dims, n_heads, d_feedforward, dropout_keep):
        self.rmsnorm1 = RMSNorm(emb_dims)
        self.attention = MultiHeadAttention(n_heads, emb_dims, d_model, d_model)
        self.dropout1 = Dropout(dropout_keep)
        self.rmsnorm2 = RMSNorm(emb_dims)
        self.fc1 = Linear(emb_dims, d_feedforward)
        self.activation = GELU()
        self.fc2 = Linear(d_feedforward, emb_dims)
        self.dropout2 = Dropout(dropout_keep)

    def forward(self, X, is_training=True):
        self.rmsnorm1.forward(X)
        self.attention.forward(self.rmsnorm1.output)
        self.dropout1.forward(self.attention.output, is_training)
        h = self.dropout1.output + X
        self.rmsnorm2.forward(h)
        self.fc1.forward(self.rmsnorm2.output)
        self.activation.forward(self.fc1.output)
        self.fc2.forward(self.activation.output)
        self.dropout2.forward(self.fc2.output, is_training)
        self.output = self.dropout2.output + h

    def backward(self, dvals):
        self.dropout2.backward(dvals)
        self.fc2.backward(self.dropout2.dinputs)
        self.activation.backward(self.fc2.dinputs)
        self.fc1.backward(self.activation.dinputs)
        self.rmsnorm2.backward(self.fc1.dinputs)
        dh = self.rmsnorm2.dinputs + dvals
        self.dropout1.backward(dh)
        self.attention.backward(self.dropout1.dinputs)
        self.rmsnorm1.backward(self.attention.dinputs)
        self.dinputs = self.rmsnorm1.dinputs + dh
