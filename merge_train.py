import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from MergeModel import MergeModel
from networks import AlexNet
import seaborn as sn
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0))


train_dataset = torchvision.datasets.ImageFolder(
    root='D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/train_data/',
    transform=transforms.Compose([
        transforms.Resize([800,800]),
        transforms.ToTensor()
    ])
)

test_dataset = torchvision.datasets.ImageFolder(
    root='D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/test_data/',
    transform=transforms.Compose([
        transforms.Resize([800,800]),
        transforms.ToTensor()
    ])
)

# Hyper parameters
num_epochs = 5
num_classes = 2
batch_size = 8
learning_rate = 0.00001
#checkpoint_interval = 1
# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


model1 = AlexNet(num_classes).to(device)
checkpoint = torch.load('./models/checkpoint_2_epoch.pth')
model1.load_state_dict(checkpoint['model_state_dict'])


model2 = AlexNet(num_classes).to(device)
checkpoint = torch.load('./origin_models/checkpoint_11_epoch.pth')
model2.load_state_dict(checkpoint['model_state_dict'])
# model.load_state_dict(torch.load('./checkpoints/model.pth'))

MM = MergeModel(model1, model2).to(device)

# model = AlexNet(num_classes).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(MM.parameters(), lr=learning_rate)

tb = SummaryWriter('./logs')


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = MM(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        tb.add_scalar('MergeLoss{}'.format(epoch+1), loss.item(), i + 1)


    checkpoint = {"model_state_dict": MM.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}
    path_checkpoint = "./merge_models/checkpoint_{}_epoch.pth".format(epoch)
    torch.save(checkpoint, path_checkpoint)
#tensorbard --logdir='./logs'