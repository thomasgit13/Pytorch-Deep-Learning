import torch 
from torch import nn 
from torch.nn import functional as F 

torch.manual_seed(101) 

class SelfAttentionHead(nn.Module):
    def __init__(self,context_size,emb_size,head_size,p):
        super(SelfAttentionHead,self).__init__()
        self.context_size = context_size 
        self.emb_size = emb_size 
        self.head_size = head_size 
        self.queries = nn.Linear(emb_size,head_size)
        self.keys = nn.Linear(emb_size,head_size)
        self.values = nn.Linear(emb_size,head_size) 
        self.dropout = nn.Dropout(p)
        self.register_buffer('mask',torch.tril(torch.ones(context_size,context_size))) 

    def forward(self,x): 
        # shape of x : (b,t,emb) 
        q = self.queries(x)  # (b,t,head_size)
        k = self.keys(x) 
        v = self.values(x)  # (b,t,head_size)

        # computing attention scores 
        weights = q.matmul(k.mT)*self.head_size**(-0.5) # (b,t,t) 
        weights = weights.masked_fill(self.mask == 0 ,float('-inf')) # (b,t,t) 
        weights = F.softmax(weights,dim =-1)
        weights = self.dropout(weights) #(b,t,t) 
        single_head_output = weights.matmul(v)
        return single_head_output

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,context_size,emb_size,head_size,p):
        super(MultiHeadAttention,self).__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(context_size,emb_size,head_size,p) for _ in range(num_heads)]
            )
        self.dropout = nn.Dropout(p) 
        self.proj = nn.Linear(emb_size,emb_size)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim =-1) # batch_size,context_size,emb_size 
        out = self.dropout(self.proj(out)) # batch_size,context_size,emb_size
        return out 


class FeedForward(nn.Module):
    def __init__(self,emb_size,p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size,4*emb_size),
            nn.ReLU(),
            nn.Linear(4*emb_size,emb_size),
            nn.Dropout(p) 
        )
    def forward(self,x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self,emb_size,n_head,context_size,p):
        super().__init__()
        head_size = emb_size // n_head 
        self.sa = MultiHeadAttention(n_head,context_size,emb_size,head_size,p)
        self.ffwd = FeedForward(emb_size,p) 
        self.ln1 = nn.LayerNorm(emb_size) 
        self.ln2 = nn.LayerNorm(emb_size) 
    def forward(self,x):
        x = x+self.sa(self.ln1(x)) 
        x = x+self.ffwd(self.ln2(x)) 
        return x 
    
token_size = 10 
embedding_dim = 128 
batch_size = 64
drop_ratio = 0.2 

x = torch.randn((batch_size,token_size,embedding_dim)) # shape : (64,10,128) - (batch_size,context_size,emb_size) 
transformer_block = TransformerBlock(emb_size=embedding_dim,n_head=4,context_size=token_size,p = drop_ratio)

out = transformer_block(x)



