import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self,no_tokens,embedding_dim):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(no_tokens,embedding_dim)
    def forward(self,x):

        # x has the indexes of tokens
        # output should have the embeddings of corresponding indexes 
        return self.embedding(x) 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dropout=0.3):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        Q_K_matmul = torch.matmul(Q, K.permute(0, 1, 3, 2))
        scores = Q_K_matmul/m.sqrt(self.d)
        # shape(scores) = [B x num_heads x Q_len x KV_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        # shape(attention_weights) = [B x num_heads x Q_len x KV_len]

        output = torch.matmul(attention_weights, V)
        # shape(output) = [B x num_heads x Q_len x D/num_heads]

        return output, attention_weights

    def forward(self, pre_q, pre_k, pre_v, mask=None):
        # shape(x) = [B x seq_len x D]

        Q = self.linear_Q(pre_q)
        K = self.linear_K(pre_k)
        V = self.linear_V(pre_v)
        # shape(Q) = [B x seq_len x D] (if in encoder, seq_len = SRC_seq_len; if in decoder, seq_len = TRG_seq_len)
        # shape(K, V) = [B x seq_len x D] (always SRC_seq_len unless in masked-multihead-attention)

        batch_size = pre_q.shape[0]

        Q = Q.reshape(batch_size, self.num_heads, -1, self.d)
        K = K.reshape(batch_size, self.num_heads, -1, self.d)
        V = V.reshape(batch_size, self.num_heads, -1, self.d)
        # shape(Q) = [B x num_heads x seq_len x D]
        # shape(K, V) = [B x num_heads x seq_len x D]

        # run scaled_dot_product_attention
        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        # shape(output) = [B x num_heads x Q_len x D/num_heads]
        # shape(attn_weights) = [B x num_heads x Q_len x KV_len]

        output = output.reshape(batch_size, -1, self.d_model)
        # shape(output) = [B x seq_len x D]

        projection = self.dropout(self.mha_linear(output))

        return projection, attn_weights


# example 
# embedding = Embedding(no_tokens=1000,embedding_dim=256) 
# three sentence, each sentence has 4 tokens
# inp_tokens = torch.tensor([[3,54,25,87],[45,2,1,47],[88,4,75,60]])  
# q = embedding(inp_tokens) 
# k = embedding(inp_tokens)
# v = embedding(inp_tokens)
# 
# mha = MultiHeadAttention(d_model=256, num_heads=2) 
# projection,attn_weights = mha(q, k, v) 
# 
# print(q.size()) => torch.Size([3, 4, 256])
# print(projection.size())  # same dimension of mha inputs  => torch.Size([3, 4, 256])
# print(attn_weights.size())  => torch.Size([3, 2, 4, 4]) 
# 

