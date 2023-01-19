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
        self.d = d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # creating multiple queries/keys/values 
        self.linear_Qs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model, self.d) for _ in range(num_heads)])

        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # shape(Q) = [B x seq_len x D/num_heads]
        # shape(K, V) = [B x seq_len x D/num_heads]

        Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        scores = Q_K_matmul/m.sqrt(self.d)
        # shape(scores) = [B x seq_len x seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        # shape(attention_weights) = [B x seq_len x seq_len]

        output = torch.matmul(attention_weights, V)
        # shape(output) = [B x seq_len x D/num_heads]

        return output, attention_weights

    def forward(self, pre_q, pre_k, pre_v, mask=None):
        # shape(x) = [B x seq_len x D]

        Q = [linear_Q(pre_q) for linear_Q in self.linear_Qs]
        K = [linear_K(pre_k) for linear_K in self.linear_Ks]
        V = [linear_V(pre_v) for linear_V in self.linear_Vs]
        # shape(Q, K, V) = [B x seq_len x D/num_heads] * num_heads

        output_per_head = []
        attn_weights_per_head = []
        # shape(output_per_head) = [B x seq_len x D/num_heads] * num_heads
        # shape(attn_weights_per_head) = [B x seq_len x seq_len] * num_heads
        
        for Q_, K_, V_ in zip(Q, K, V):
            print(f'query shape = {Q_.size()}')
            ##run scaled_dot_product_attention
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_, mask)
            # shape(output) = [B x seq_len x D/num_heads]
            # shape(attn_weights_per_head) = [B x seq_len x seq_len]
            output_per_head.append(output) 
            attn_weights_per_head.append(attn_weight)

            print(f'output shape = {output.size()}') 
            print(f'attention weights  shape = {attn_weight.size()}') 

        output = torch.cat(output_per_head, -1)
        print(f'final output shape = {output.size()}') 

        # actual stacking dimension : (no_of_items,item_dim)
        # we need to make the dimension as follows ( batch_size,no_of_items,item_sub_dimension)
        attn_weights = torch.stack(attn_weights_per_head).permute(1, 0, 2, 3)
        print(f'final attn_weights shape = {attn_weights.size()}')

        # shape(output) = [B x seq_len x D]
        # shape(attn_weights) = [B x num_heads x seq_len x seq_len]
        
        projection = self.dropout(self.mha_linear(output))
        print(f'projection shape = {projection.size()}')
        return projection, attn_weights

# example 
# embedding = Embedding(no_tokens=1000,embedding_dim=256) 
# 
# inp_tokens = torch.tensor([[3,54,25,87],[45,2,1,47],[88,4,75,60]]) # three sentence, each sentence has 4 tokens 
# q = embedding(inp_tokens) 
# k = embedding(inp_tokens)
# v = embedding(inp_tokens)
# 
# mha = MultiHeadAttention(d_model=256, num_heads=2) 
# projection,attn_weights = mha(q, k, v) 

# query shape = torch.Size([3, 4, 128])
# output shape = torch.Size([3, 4, 128])
# attention weights  shape = torch.Size([3, 4, 4])
# query shape = torch.Size([3, 4, 128])
# output shape = torch.Size([3, 4, 128])
# attention weights  shape = torch.Size([3, 4, 4])
# final output shape = torch.Size([3, 4, 256])
# final attn_weights shape = torch.Size([3, 2, 4, 4])
# projection shape = torch.Size([3, 4, 256])

