import torch 
import torch.nn as nn
from residual_layer_norm import ResidualLayerNorm
from MultiHeadAttention import MultiHeadAttention
from efficient_mha import MultiHeadAttention as EfficientMultiHeadAttention
from pwffn import PWFFN



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3, efficient_mha=False):
        super().__init__()

        # initalize these
        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)

        if efficient_mha:
            self.mha = EfficientMultiHeadAttention(d_model, num_heads, dropout)
        else:
            self.mha = MultiHeadAttention(d_model, num_heads, dropout)

        self.ff = PWFFN(d_model, d_ff, dropout)  # output of ff layer should be of size d_model

    def forward(self, x, mask): 
        # D represents the model dim ( embedding dim) 
        # shape(x) = [B x seq_len x D]

        mha, encoder_attention_weights = self.mha(x, x, x, mask)
        # shape(mha) = [B x seq_len x D]
        # shape(encoder_attention_weights) = [B x num_heads x seq_len x seq_len]

        norm1 = self.norm_1(mha, x)
        # shape(norm1) = [B x seq_len x D]

        ff = self.ff(norm1)
        norm2 = self.norm_2(ff, norm1)
        # shape(ff) = [B x seq_len x D]
        # shape(norm2) = [B x seq_len x D]

        return norm2, encoder_attention_weights

# example 
class Embedding(nn.Module):
    def __init__(self,no_tokens,embedding_dim):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(no_tokens,embedding_dim)
    def forward(self,x):

        # x has the indexes of tokens
        # output should have the embeddings of corresponding indexes 
        return self.embedding(x) 

# EXAMPLE 
# embedding = Embedding(no_tokens=1000,embedding_dim=256) 
# 
# inp_tokens = torch.tensor([[3,54,25,87],[45,2,1,47],[88,4,75,60]]) # three sentence, each sentence has 4 tokens 
# x = embedding(inp_tokens) 
# 
# enc_layer = EncoderLayer(d_model=256,num_heads = 8,d_ff = 16,efficient_mha = True)
# 
# encoder_output,encoder_attention_weights = enc_layer(x,mask = None)
# 
# print(encoder_output.size())
# print(encoder_attention_weights.size())

