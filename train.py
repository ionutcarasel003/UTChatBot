import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from config import *
from preprocessing import load_data, build_vocab, tokenize
from dataset import QADataset
from model import Seq2SeqLSTM

device = torch.device("cuda"
                      ""
                      "" if torch.cuda.is_available() else "cpu")
print(device)

# Load and prepare data
pairs = load_data()
word2idx, idx2word = build_vocab(pairs)
dataset = QADataset(pairs, word2idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
vocab_size = len(word2idx)
model = Seq2SeqLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for q, a in dataloader:
        output = model(q, a[:, :-1])
        loss = criterion(output.view(-1, vocab_size), a[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "chatbot_model.pt")
