import torch 
from torch import nn 
import torch.nn.functional as F
import math 

class Embedding(nn.Module):
    def __init__(self,no_tokens,embedding_dim):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(no_tokens,embedding_dim)
    def forward(self,x):

        # x has the indexes of tokens
        # output should have the embeddings of corresponding indexes 
        return self.embedding(x) 

# example 
# we have 1000 tokens in the training corpus,and embedding dimension is 4 
embedding_layer = Embedding(1000,4)

# sample index tensor 
test_input = torch.tensor([[23,38,42,4,3,29,99,100],[4,2,53,23,50,23,58,56]]) 
output = embedding_layer(test_input)
 
print(f'output after embedding   = {output.size()}') 

# implementation of single head 
class SelfAttention(nn.Module):
    def __init__(self,dim_e,dim_q,dim_k,dim_v):
        self.dim = dim_q 
        # dim_e : embedding dimension 
        # dim_q : query dimension 
        # dim_k : key dimension 
        # dim_v : value dimension 
        super(SelfAttention,self).__init__()
        self.queries = nn.Linear(dim_e,dim_q) 
        self.keys = nn.Linear(dim_e,dim_k) 
        self.values = nn.Linear(dim_e,dim_v) 
    
    def forward(self,x):
        # initializing the queries/keys/values 
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        qk = torch.matmul(queries,keys.mT) 
        qk_scaled = qk/math.sqrt(self.dim)
        att_weights = F.softmax(qk_scaled,dim = 1)

        output = torch.matmul(att_weights,values)
        return output,att_weights 

        # x contains the embedding vector representation of each token
        # shape of x : (batch_size,no_of_tokens,embedding_dim)


self_attention_layer = SelfAttention(dim_e = 4,dim_q = 6,dim_k = 6,dim_v = 6)
output,att_weights = self_attention_layer(output)

print(f'output dimension : {output.size()}')
print(f'attention weights dimension : {att_weights.size()}')

print(output)
print(att_weights)


