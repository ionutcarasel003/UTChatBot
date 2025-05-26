from config import MAX_LEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

def encode(sentence, word2idx):
    tokens = [word2idx.get(word, 0) for word in sentence]
    tokens = [word2idx[SOS_TOKEN]] + tokens[:MAX_LEN - 2] + [word2idx[EOS_TOKEN]]
    tokens += [word2idx[PAD_TOKEN]] * (MAX_LEN - len(tokens))
    return tokens

def decode(indices, idx2word):
    words = [idx2word.get(idx, "") for idx in indices]
    return " ".join(w for w in words if w not in ["<PAD>", "<SOS>", "<EOS>"])
