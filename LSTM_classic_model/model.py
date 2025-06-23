import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, target_seq):
        embed_in = self.embedding(input_seq)
        _, (hidden, cell) = self.encoder(embed_in)

        embed_out = self.embedding(target_seq)
        output, _ = self.decoder(embed_out, (hidden, cell))
        return self.fc(output)
