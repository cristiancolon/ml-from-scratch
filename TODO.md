To-Do

Must-have:
* Final RMSNorm + output layer
  - Applied once in the model class after the last block: RMSNorm(emb_dims), then Linear(emb_dims, vocab_size, use_bias=False)
  - Targets are the input tokens shifted left by one (position t predicts token t+1)
  - When chatting, only the last position's logits are needed
* Model class: embedding, blocks, final norm, output layer (new class)
  - forward(ids, is_training): embedding -> blocks -> final norm -> output layer -> combined loss
  - backward: same path in reverse, ending at the embedding
  - One flat list of every layer with weights, shared by the optimizer, gradient clipping and checkpoints
  - clear_cache() for every block; save/load weights + Adam state + step count (resuming, and the start of SFT)
* Weight init scaled by fan-in + scaled-down residual output projections
  - Linear std = 1/sqrt(n_inputs) instead of 0.1, plus an optional extra scale factor
  - MHA combination and fc2 get an extra 1/sqrt(2 * n_layers), so the block needs the total layer count
  - Output layer std ~0.02 so the starting loss is close to ln(vocab_size) ~ 10.4
  - Check: stream RMS stays ~1 through all blocks
* Learning-rate warmup + decay
  - A function of the step that sets optimizer.lr before each update; Adam's decay argument stays 0
  - Linear warmup from ~0 to peak over the first ~1-2% of steps
  - Cosine or linear decay to ~10% of peak by the last step, so total steps must be known up front
  - Pick the peak with short local runs; smaller batches usually want a lower peak
* Gradient clipping
  - Between backward and the optimizer step
  - Global norm: square root of the sum of squares of every dweights and dbiases across the whole model
  - If it's above 1.0, scale every gradient by 1.0 / norm
  - Log the norm each step; spikes are an early warning

SFT:
* Special tokens in BPE: user/assistant role markers, end-of-turn, end-of-document
  - IDs right after the last merge; save them with the vocab and merges
  - Plain-text encode never produces them; the chat template inserts their IDs directly
  - Add before tokenizing ClimbMix: end-of-document separates concatenated documents
  - Embedding and output layer sized to the full vocab including them
* Chat template: one fixed format for training and chatting
  - Conversation (list of role + text) -> token IDs + mask
  - Each turn: role marker, text, end-of-turn
  - To chat: template the history, append the assistant marker, sample until end-of-turn
* Conversation loader: pack/pad to fixed length, with the mask
  - Template each conversation; truncate or drop ones longer than 1024 tokens
  - Pack several per row (separated by end-of-document) or pad; pad positions get mask 0
  - Shift targets and mask left by one, same as pretraining; shuffle each epoch
* Loss masking in combined loss: only assistant tokens count
  - Optional mask argument (1 = assistant token, including its end-of-turn so it learns to stop)
  - Forward: average loss over mask-1 positions only
  - Backward: zero the gradient where mask is 0; n = number of mask-1 positions
  - No mask = pretraining behavior (every position counts)

Nice-to-have:
* SwiGLU (new classes: SiLU, feed-forward)
* AdamW (new class)
* Tie output layer to embedding
