from collections import deque, Counter
from bidict import bidict
import json


class BPE:
    def __init__(self):
        self.vocab = bidict()
        self.bpe_merges = {}

    
    def find_top_pair(self, token_ids):
        pairs = Counter(zip(token_ids, token_ids[1:]))
        return max(pairs.items(), key=lambda x: x[1])[0] if pairs else None

    
    def replace_pair(self, token_ids, target_pair_id, new_pair_id):
        dq = deque(token_ids)
        replaced = []
        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == target_pair_id:
                replaced.append(new_pair_id)
                dq.popleft()
            else:
                replaced.append(current)
        return replaced
    
    
    def train(self, corpus, vocab_size):
        self.vocab = bidict({i: bytes([i]) for i in range(256)})
        self.bpe_merges = {}
        token_ids = list(corpus.encode("utf-8"))

        for new_pair_id in range(len(self.vocab), vocab_size):
            target_pair_id = self.find_top_pair(token_ids)
            if target_pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, target_pair_id, new_pair_id)
            self.bpe_merges[target_pair_id] = new_pair_id
        for (p0, p1), new_pair_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_pair_id] = merged_token


    def encode(self, corpus):
        token_ids = list(corpus.encode("utf-8"))
        for pair, merged_id in self.bpe_merges.items():
            token_ids = self.replace_pair(token_ids, pair, merged_id)
        return token_ids


    def decode(self, token_ids):
        tokens = b"".join(self.vocab[token_id] for token_id in token_ids)
        return tokens.decode("utf-8")


    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        serialized_vocab = {
            str(token_id): token_bytes.hex() for token_id, token_bytes in self.vocab.items()
        }
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(serialized_vocab, file, indent=2)
        merges_list = [
            {"pair": list(pair), "new_id": new_id} for pair, new_id in self.bpe_merges.items()
        ]
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            json.dump(merges_list, file, indent=2)


    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
        self.vocab = bidict({
            int(token_id): bytes.fromhex(token_bytes) for token_id, token_bytes in loaded_vocab.items()
        })
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
        self.bpe_merges = {
            tuple(merge["pair"]): merge["new_id"] for merge in merges_list 
        }
