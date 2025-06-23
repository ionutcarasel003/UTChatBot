import torch
from transformer_model import TransformerChatBot
from transformer_dataset import Tokenizer
from transformer_config import *
import json

with open(DATA_PATH, "r", encoding="utf-8") as f:
    qa_pairs = [(item["question"], item["answer"]) for item in json.load(f)]

tokenizer = Tokenizer(qa_pairs)
vocab_size = len(tokenizer.word2idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerChatBot(vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def generate_response(model, tokenizer, question, max_len=MAX_LEN):
    model.eval()
    src = torch.tensor([tokenizer.encode(question, max_len)], dtype=torch.long).to(device)
    tgt = torch.tensor([[tokenizer.word2idx[SOS_TOKEN]]], dtype=torch.long).to(device)

    for _ in range(max_len):
        tgt_mask = model.generate_sqr_subsequent_mask(tgt.size(1)).to(device)
        out = model(src, tgt, tgt_mask=tgt_mask)

        probs = torch.softmax(out[:, -1, :], dim=-1)
        topk = torch.topk(probs, k=10, dim=-1)
        indices = topk.indices[0]
        values = topk.values[0]
        next_token = indices[torch.multinomial(values, 1)].unsqueeze(0)

        tgt = torch.cat([tgt, next_token], dim=1)

        # Dacă e <EOS>, oprim
        if next_token.item() == tokenizer.word2idx[EOS_TOKEN]:
            break

    decoded = tokenizer.decode(tgt[0].tolist())
    return decoded

print("Chatbot is ready! Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.lower().strip() == "exit":
        break
    response = generate_response(model, tokenizer, question)
    print("Bot:", response)