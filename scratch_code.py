import torch 
import torch.nn as nn 
torch.manual_seed(101) 

linear_layer = nn.Linear(8,10) 
inp_train = torch.randn(size = (3,5,8))

train_out = linear_layer(inp_train)

inp_test = torch.randn(size = (1,1,8))
test_out = linear_layer(inp_test)



