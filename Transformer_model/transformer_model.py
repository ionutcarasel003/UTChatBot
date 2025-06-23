import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_length = 5000):
        super().__init__()
        # position encoding
        pe = torch.zeros(max_length,d_model)
        pos = torch.arange(0,max_length,dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe
    def forward(self,x):
        x += self.pe[:, :x.size(1)]
        return x

class TransformerChatBot(nn.Module):
    def __init__(self,vocab_size, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pe = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_layers,
            num_decoder_layers = num_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first = True
        )
        self.fc_out = nn.Linear(d_model,vocab_size)
        self.d_model = d_model

    def generate_sqr_subsequent_mask(self,dec_size):
        mask = (torch.triu(torch.ones((dec_size,dec_size))) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pe(src)
        tgt = self.pe(tgt)

        if tgt_mask is None:
            tgt_mask = self.generate_sqr_subsequent_mask(tgt.size(1)).to(tgt.device)

        out = self.transformer(
            src,
            tgt,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_key_padding_mask,
            tgt_key_padding_mask = tgt_key_padding_mask
        )
        return self.fc_out(out)


