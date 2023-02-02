import torch 
import os 
import torchvision 
import torchvision.transforms as transforms
import torchvision.models as models 
import torch.optim as optim 
import torch.nn as nn

TRAIN_IMAGE_DIR = 'Images/Training_Images'
TEST_IMAGE_DIR = 'Images/Testing_Images' 

mean = torch.tensor([0.4636, 0.4570, 0.4486])
std = torch.tensor([0.2284, 0.1945, 0.1807])

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean,std) 
])

test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std) 
])


train_dataset = torchvision.datasets.ImageFolder(
    root=TRAIN_IMAGE_DIR,transform = train_transform)

test_dataset = torchvision.datasets.ImageFolder(
    root=TEST_IMAGE_DIR,transform = test_transform)


train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,batch_size = 32,shuffle = False)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,batch_size = 32,shuffle = False)

# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# print(labels)

def get_mean_std(loader):
    mean = 0 
    std = 0 
    image_count = 0 
    for images,_ in loader:
        B,C,W,H = images.shape
        images = images.view(B,C,-1) 
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0) 
        image_count+=B
    mean /=image_count
    std /=image_count

    return mean,std,image_count 


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

resnet18_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
num_features = resnet18_model.fc.in_features
num_classes = 4
resnet18_model.fc = nn.Linear(num_features,num_classes) 
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(),lr = 0.001,momentum=0.9,weight_decay=0.003)

def evaluate_model_on_test_set(model,test_loader):
    model.eval()
    predicted_correct = 0 
    total = 0 
    with torch.no_grad():
        for data in test_loader:
            images,labels = data 
            images = images.to(device)
            labels = labels.to(device)
            total +=images.size(0)

            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            predicted_correct +=(predicted == labels).sum().item()
    epoch_acc = 100.0* predicted_correct/total 
    print(f'Validation : No of correct predictions : {predicted_correct} \
        ; epoch accuaracy : {epoch_acc}')


def train_nn(model,train_loader,test_loader,criterion,optimizer,n_epochs):
    for epoch in range(n_epochs):
        print(f'Epoch = {epoch}')
        model.train() 
        running_loss = 0.0 
        running_correct = 0.0 
        total = 0 
        for data in train_loader:
            images,labels = data
            images = images.to(device) 
            labels = labels.to(device)
            total += labels.size(0) 

            optimizer.zero_grad()
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            loss = criterion(outputs,labels) 

            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            running_correct += (labels ==predicted).sum().item()
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct/total 
        print(f'No of correct predictions : {running_correct} ; epoch accuaracy : {epoch_acc} ; \
            epoch_loss: {epoch_loss}')

        evaluate_model_on_test_set(model,test_loader) 
    print('Finished')
    return model 
n_epochs = 3 
model = train_nn(resnet18_model,train_loader,test_loader,loss_fn,optimizer,n_epochs)
