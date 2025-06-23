import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformer_config import *
from transformer_model import TransformerChatBot
from transformer_dataset import Tokenizer, TransformerDataset
import json
import matplotlib.pyplot as plt

best_loss = float('inf')

# Load and tokenize data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    pairs = [(item["question"], item["answer"]) for item in json.load(f)]

tokenizer = Tokenizer(pairs)
dataset = TransformerDataset(pairs, tokenizer, max_len=MAX_LEN)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vocab_size = len(tokenizer.word2idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerChatBot(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=LR)

losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_tokens = 0

    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Boolean masks
        src_key_padding_mask = (src == tokenizer.word2idx[PAD_TOKEN])
        tgt_key_padding_mask = (tgt_input == tokenizer.word2idx[PAD_TOKEN])

        # Debug shape checks
        assert src.shape == src_key_padding_mask.shape, "Mask shape mismatch for src"
        assert tgt_input.shape == tgt_key_padding_mask.shape, "Mask shape mismatch for tgt"

        logits = model(src, tgt_input,
                       src_key_padding_mask=src_key_padding_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask).log_softmax(-1)

        logits = logits.view(-1, vocab_size)
        tgt_output = tgt_output.reshape(-1)

        loss_per_token = criterion(logits, tgt_output)
        tokens_nonpad = (tgt_output != tokenizer.word2idx[PAD_TOKEN]).sum()
        loss = loss_per_token.sum() / tokens_nonpad

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * tokens_nonpad.item()
        total_tokens += tokens_nonpad.item()

    epoch_loss = total_loss / total_tokens
    losses.append(epoch_loss)
    with open("loss_log.csv", "a") as f:
        f.write(f"{epoch + 1},{epoch_loss:.6f}\n")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\nEpoch {epoch + 1}/{EPOCHS} ✅ Model salvat cu loss: {epoch_loss:.6f}")
    else:
        print(f"\nEpoch {epoch + 1}/{EPOCHS} Loss: {epoch_loss:.6f}")

plt.plot(range(1, EPOCHS + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.savefig("transformer_training_loss.png")
plt.show()
