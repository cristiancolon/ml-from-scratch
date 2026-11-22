import numpy as np
from block import Transformer
from layer import Linear, RMSNorm, Embedding
from loss import Softmax_CategoricalCrossEntropy
from optimizer import Adam

class HGPT:
    def __init__(self, vocab_size, n_layers, d_model, emb_dims, n_heads, d_feedforward, dropout_keep):
        self.embedding = Embedding(vocab_size, emb_dims, 0.02)
        self.blocks = [Transformer(d_model, emb_dims, n_heads, d_feedforward, dropout_keep) for _ in range(n_layers)]
        self.norm = RMSNorm(emb_dims)
        self.output = Linear(emb_dims, vocab_size, use_bias=False)
        self.loss = Softmax_CategoricalCrossEntropy()

        # Every layer with weights, shared by the optimizer, gradient clipping and checkpoints
        self.layers = [self.embedding]
        for b in self.blocks:
            self.layers += [b.rmsnorm1]
            for h in b.attention.heads:
                self.layers += [h.query_w, h.key_w, h.value_w]
            self.layers += [b.attention.combination, b.rmsnorm2, b.fc1, b.fc2]
        self.layers += [self.norm, self.output]

    def forward(self, ids, targets=None, is_training=True):
        self.embedding.forward(ids)
        X = self.embedding.output
        for b in self.blocks:
            b.forward(X, is_training)
            X = b.output
        self.norm.forward(X)
        if targets is None:
            self.output.forward(self.norm.output[..., -1, :])
            return self.output.output
        self.output.forward(self.norm.output)
        return self.loss.forward(self.output.output, targets)

    def backward(self, targets):
        self.loss.backward(self.loss.output, targets)
        self.output.backward(self.loss.dinputs)
        self.norm.backward(self.output.dinputs)
        dvals = self.norm.dinputs
        for b in reversed(self.blocks):
            b.backward(dvals)
            dvals = b.dinputs
        self.embedding.backward(dvals)

    def clear_cache(self):
        for b in self.blocks:
            b.attention.clear_cache()

    def save(self, path, optimizer):
        arrays = {'step': np.array(optimizer.iterations)}
        for i, l in enumerate(self.layers):
            arrays[f'{i}_w'] = l.weights
            if l.biases is not None:
                arrays[f'{i}_b'] = l.biases
            for j, (m, c) in enumerate(zip(getattr(l, 'momentums', []), getattr(l, 'caches', []))):
                arrays[f'{i}_m{j}'], arrays[f'{i}_c{j}'] = m, c
        np.savez(path, **arrays)

    def load(self, path, optimizer=None):
        # Pass the optimizer to resume training (Adam state + step); leave it out to load weights only.
        d = np.load(path)
        for i, l in enumerate(self.layers):
            l.weights = d[f'{i}_w']
            if l.biases is not None:
                l.biases = d[f'{i}_b']
            if optimizer is not None and f'{i}_m0' in d:
                n = 2 if l.biases is not None else 1
                l.momentums = [d[f'{i}_m{j}'] for j in range(n)]
                l.caches = [d[f'{i}_c{j}'] for j in range(n)]
        if optimizer is not None:
            optimizer.iterations = int(d['step'])
