import torch 
import os 
import torchvision 
import torchvision.transforms as transforms

IMAGE_DIR = 'weather_data'
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
train_dataset = torchvision.datasets.ImageFolder(
    root=IMAGE_DIR,transform = train_transform)
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,batch_size = 32,shuffle = False)

# print(train_loader)

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
mean,std,image_count = get_mean_std(train_loader)
print(mean,std,image_count) 

# a = torch.randn(size = (32,3,48))
# print(a.mean(2).sum(0).shape)

#  mean = tensor([0.4636, 0.4570, 0.4486])
#  std = tensor([0.2284, 0.1945, 0.1807])


