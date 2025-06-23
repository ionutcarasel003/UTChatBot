import torch
from nltk.corpus import words
from torch.utils.data import Dataset
import re
from transformer_config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
class Tokenizer:
    def __init__(self,pairs, min_freq = 1):
        self.word2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        self.idx2word = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN}
        self.freq = {}
        self.build_vocab(pairs, min_freq)

    def build_vocab(self, pairs, min_freq):
        idx = len(self.word2idx)
        for q, a in pairs:
            for sentence in [q,a]:
                for word in self.tokenize(sentence):
                    self.freq[word] = self.freq.get(word,0) + 1
                    if self.freq[word] == min_freq:
                        self.word2idx[word] = idx
                        self.idx2word[idx] = word
                        idx += 1

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def encode(self, text, max_len):
        tokens = [SOS_TOKEN] + self.tokenize(text) + [EOS_TOKEN]
        ids = [self.word2idx.get(token, self.word2idx[PAD_TOKEN])for token in tokens]
        return ids[:max_len] + [self.word2idx[PAD_TOKEN]] * (max_len - len(ids))

    def decode(self,ids):
        words = [self.idx2word.get(i,PAD_TOKEN) for i in ids]
        return " ".join([w for w in words if w not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]])

class TransformerDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len = 30):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        q_ids = torch.tensor(self.tokenizer.encode(q, self.max_len), dtype=torch.long)
        a_ids = torch.tensor(self.tokenizer.encode(a, self.max_len), dtype=torch.long)
        assert q_ids.size(0) == self.max_len, "Encoded question length mismatch"
        assert a_ids.size(0) == self.max_len, "Encoded answer length mismatch"
        return q_ids, a_ids