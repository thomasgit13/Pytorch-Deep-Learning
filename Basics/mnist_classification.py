import torch
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from tqdm import tqdm 
from torch.utils.data import (
    DataLoader,
) 

torch.manual_seed(seed = 101)
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
    
        self.fc1 = nn.Linear(input_size, 50) # 784 nodes to 50
        self.fc2 = nn.Linear(50, num_classes) # 50 to 10 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load MNIST Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def check_accuracy(loader,model):
    num_correct = 0 
    num_samples = 0 
    model.eval() 
    with torch.inference_mode():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device) 
            x = x.reshape(x.shape[0],-1) 
            scores = model(x) 
            _,predictions = scores.max(1) 
            num_correct +=(predictions==y).sum() 
            num_samples += predictions.size(0) 
    model.train()
    return num_correct/num_samples

# Train Network
for epoch in range(num_epochs):
    num_correct = 0 
    num_samples = 0 
    for batch_idx, (data, targets) in enumerate(train_loader):
        model.train() 
        data = data.to(device=device) # [64,1,28,28]
        targets = targets.to(device=device) 
        data = data.reshape(data.shape[0], -1) # [64,784] 
        
        scores = model(data)
        loss = criterion(scores,targets) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # after each epoch print the accuracy of train and test 
    print(f'Epoch = {epoch} ;   Train Accuracy = {check_accuracy(train_loader,model)*100:.2f}',end = ' ; ') 
    print(f'Test Accuracy = {check_accuracy(test_loader,model)*100:.2f}') 

print(f'Trained Model : Final Train Accuracy = {check_accuracy(train_loader,model)*100:.2f}',end = ' ; ') 
print(f'Final Test Accuracy =  {check_accuracy(test_loader,model)*100:.2f}')
