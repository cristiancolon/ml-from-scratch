import numpy as np
from block import Transformer
from layer import Linear, RMSNorm, Embedding, Dropout
from loss import Softmax_CategoricalCrossEntropy
from optimizer import Adam

class CGPT:
    def __init__(self, vocab_size, n_layers, d_model, embed_dims, n_heads, d_feedforward, dropout_keep):
        self.embedding = Embedding(vocab_size, embed_dims, 0.02)
        self.dropout = Dropout(dropout_keep)
        self.transformers = [Transformer(d_model, embed_dims, n_heads, d_feedforward, dropout_keep) for _ in range(n_layers)]
        self.rmsnorm = RMSNorm(embed_dims)
        self.linear = Linear(embed_dims, vocab_size, use_bias=False)
        self.loss = Softmax_CategoricalCrossEntropy()

        self.w_layers = [self.embedding, self.linear, self.rmsnorm]
        for t in self.transformers:
            self.w_layers += [t.rmsnorm1]
            for h in t.attention.heads:
                self.w_layers += [h.query_w, h.key_w, h.value_w]
            self.w_layers += [t.attention.combination, t.rmsnorm2, t.fc1, t.fc2]

    def forward(self, token_ids, targets=None, is_training=True):
