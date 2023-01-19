import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from positional_encoding import PositionalEncoding
from embed import Embeddings



class Encoder(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model,
                 num_heads, num_layers,
                 d_ff, device="cpu", dropout=0.3, efficient_mha=False):
        super().__init__()

        self.embedding = Embedding

        self.PE = PositionalEncoding(
            d_model, device=device)

        self.encoders = nn.ModuleList([EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            efficient_mha
        ) for layer in range(num_layers)])

    def forward(self, x, mask=None):
        # shape(x) = [B x SRC_seq_len]

        embeddings = self.embedding(x)
        encoding = self.PE(embeddings)
        # shape(embeddings) = [B x SRC_seq_len x D]
        # shape(encoding) = [B x SRC_seq_len x D]

        for encoder in self.encoders:
            encoding, encoder_attention_weights = encoder(encoding, mask)
            # shape(encoding) = [B x SRC_seq_len x D]
            # shape(encoder_attention_weights) = [B x SRC_seq_len x SRC_seq_len]

        return encoding, encoder_attention_weights


embed = Embeddings(vocab_size=1000,padding_idx=None,d_model = 256) 
encoder = Encoder(embed,d_model = 256,num_heads = 8,num_layers = 3,d_ff = 64,efficient_mha = True) 

x = torch.tensor([[3,42,48,2],[48,23,14,2]])

enc_output,enc_att_weights = encoder(x) 

print(enc_output.size())
print(enc_att_weights.size())

