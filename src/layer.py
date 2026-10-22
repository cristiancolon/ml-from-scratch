import numpy as np
from activation import Softmax

class Dense:
    def __init__(self, n_inputs, n_neurons, weight_l1reg=0, bias_l1reg=0, weight_l2reg=0, bias_l2reg=0, use_bias=True):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) if use_bias else None
        self.weight_l1reg = weight_l1reg
        self.bias_l1reg = bias_l1reg
        self.weight_l2reg = weight_l2reg
        self.bias_l2reg = bias_l2reg
    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X, self.weights) + self.biases if self.biases is not None else np.dot(X, self.weights) 
    def backward(self, dvals):
        inputs = self.inputs.reshape(-1, self.weights.shape[0])
        dvals = dvals.reshape(-1, self.weights.shape[1])
        self.dweights = np.dot(inputs.T, dvals)
        if self.biases is not None:
            self.dbiases = np.sum(dvals, axis=0, keepdims=True)
        self.dinputs = (np.dot(dvals, self.weights.T)).reshape(self.inputs.shape)

        d_l1 = np.ones_like(self.weights)
        d_l1[self.weights < 0] = -1
        self.dweights += self.weight_l1reg * d_l1
        if self.biases is not None:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases < 0] = -1
            self.dbiases += self.bias_l1reg * d_l1

        d_l2 = 2 * self.weights
        self.dweights += self.weight_l2reg * d_l2
        if self.biases is not None:
            d_l2 = 2 * self.biases
            self.dbiases += self.bias_l2reg * d_l2

class Dropout:
    def __init__(self, dropout_keep):
        self.dropout_keep = dropout_keep
    def forward(self, X, is_training=True):
        self.inputs = X
        self.bernoullis = np.random.binomial(1, self.dropout_keep, size=self.inputs.shape)
        self.output = self.inputs * self.bernoullis / self.dropout_keep if is_training else self.inputs
    def backward(self, dvals, is_training=True):
        self.dinputs = dvals * self.bernoullis / self.dropout_keep if is_training else dvals

class Embedding:
    def __init__(self, vocab_size, embed_dims, scale):
        self.weights = np.random.randn(vocab_size, embed_dims) * scale
        self.biases = None
        self.weight_l1reg = 0
        self.bias_l1reg = 0
        self.weight_l2reg = 0
        self.bias_l2reg = 0
    def forward(self, X):
        self.inputs = X
        self.output = self.weights[X]
    def backward(self, dvals):
        self.dweights = np.zeros_like(self.weights)
        np.add.at(self.dweights, self.inputs, dvals)

class SelfAttentionHead:
    def __init__(self, embed_dims, qk_dims, v_dims):
        self.query_w = Dense(embed_dims, qk_dims)
        self.key_w = Dense(embed_dims, qk_dims)
        self.value_w = Dense(embed_dims, v_dims)
        self.softmax = Softmax()
        self.qk_dims = qk_dims
        self.rope = RotaryPositionalEmbeddings(self.qk_dims)
    def forward(self, X):
        self.inputs = X
        self.query_w.forward(X)
        self.key_w.forward(X)
        self.value_w.forward(X)
       
        self.q = self.query_w.output
        self.k = self.key_w.output
        self.v = self.value_w.output

        self.rope.forward(self.q)
        self.q_prime = self.rope.output
        
        self.rope.forward(self.k)
        self.k_prime = self.rope.output

        self.s = self.q_prime @ self.k_prime.swapaxes(-1, -2) / np.sqrt(self.qk_dims)
        seq_len = X.shape[-2]
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        s_masked = np.where(mask, -np.inf, self.s)
        self.softmax.forward(s_masked)
        self.output = np.matmul(self.softmax.output, self.v)
        
    def backward(self, dvals):
        da = dvals @ self.v.swapaxes(-1, -2)
        dv = self.softmax.output.swapaxes(-1, -2) @ dvals

        self.softmax.backward(da)
        ds = self.softmax.dinputs / np.sqrt(self.qk_dims)

        dq_prime = ds @ self.k_prime
        self.rope.backward(dq_prime)
        dq = self.rope.dinputs

        dk_prime = ds.swapaxes(-1, -2) @ self.q_prime
        self.rope.backward(dk_prime)
        dk = self.rope.dinputs
        
        self.query_w.backward(dq)
        self.key_w.backward(dk)
        self.value_w.backward(dv)
        
        self.dinputs = self.query_w.dinputs + self.key_w.dinputs + self.value_w.dinputs

class MultiHeadAttention:
    def __init__(self, head_count, embed_dims, qk_dims, v_dims):
        self.heads = [SelfAttentionHead(embed_dims, qk_dims, v_dims) for _ in range(head_count)]
        self.combination = Dense(head_count * v_dims, embed_dims)
    def forward(self, X):
        self.inputs = X
        for head in self.heads:
            head.forward(X)
        multi_outs = np.concatenate([h.output for h in self.heads], axis=-1)
        self.combination.forward(multi_outs)
        self.output = self.combination.output
    def backward(self, dvals):
        self.combination.backward(dvals)
        dheads = np.split(self.combination.dinputs, len(self.heads), axis=-1)
        self.dinputs = np.zeros_like(self.inputs)
        for head, dhead in zip(self.heads, dheads):
            head.backward(dhead)
            self.dinputs += head.dinputs

class RotaryPositionalEmbeddings:
    def __init__(self, dims, base = 10_000):
        self.base = base
        self.dims = dims
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, X):
        thetas = np.array([self.base ** (-2 * i / self.dims) for i in range(self.dims // 2)])
        positions = np.array([i for i in range(X.shape[-2])])
        angles = np.outer(positions, thetas)
        X0, X1 = X[..., 0::2], X[..., 1::2]
        self.output = np.empty_like(X)
        outs0, outs1 = X0 * np.cos(angles) - X1 * np.sin(angles), X0 * np.sin(angles) + X1 * np.cos(angles)
        self.output[...,0::2], self.output[..., 1::2] = outs0, outs1

    def backward(self, dvals):
        thetas = np.array([self.base ** (-2 * i / self.dims) for i in range(self.dims // 2)])
        positions = np.array([i for i in range(dvals.shape[-2])])
        angles = np.outer(positions, thetas)
        dvals0, dvals1 = dvals[..., 0::2], dvals[..., 1::2]
        self.dinputs = np.empty_like(dvals)
        dinputs0, dinputs1 = dvals0 * np.cos(angles) + dvals1 * np.sin(angles), - dvals0 * np.sin(angles) + dvals1 * np.cos(angles)
        self.dinputs[...,0::2], self.dinputs[..., 1::2] = dinputs0, dinputs1
