import torch 
from torch import nn 
torch.manual_seed(101) 

m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(5, affine=False)
input = torch.randn(2,5, 3)
print(input) 
output = m(input)
