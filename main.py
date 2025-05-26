import torch
from preprocessing import load_data, build_vocab, tokenize
from model import Seq2SeqLSTM
from utils import encode, decode
from config import *

# Load everything
pairs = load_data()
word2idx, idx2word = build_vocab(pairs)
vocab_size = len(word2idx)

model = Seq2SeqLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("chatbot_model.pt"))
model.eval()

def predict(question):
    tokens = tokenize(question.lower())
    input_seq = torch.tensor(encode(tokens, word2idx)).unsqueeze(0)
    with torch.no_grad():
        embed = model.embedding(input_seq)
        _, (hidden, cell) = model.encoder(embed)

        outputs = [word2idx[SOS_TOKEN]]
        for _ in range(MAX_LEN):
            tgt = torch.tensor([outputs[-1]]).unsqueeze(0)
            tgt_embed = model.embedding(tgt)
            out, (hidden, cell) = model.decoder(tgt_embed, (hidden, cell))
            pred = model.fc(out.squeeze(1))
            next_word = pred.argmax(dim=1).item()
            if next_word == word2idx[EOS_TOKEN]:
                break
            outputs.append(next_word)

    return decode(outputs[1:], idx2word)

# Test loop
if __name__ == "__main__":
    while True:
        q = input("Tu: ")
        if q.lower() in ['exit', 'quit']:
            break
        print("Bot:", predict(q))
