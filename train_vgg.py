import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from VGG16 import VGG

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0))

train_dataset = torchvision.datasets.ImageFolder(
    root='D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/train_data',
    transform=transforms.Compose([
        transforms.Resize([800,800]),
        transforms.ToTensor()
    ])
)

test_dataset = torchvision.datasets.ImageFolder(
    root='D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/test_data',
    transform=transforms.Compose([
        transforms.Resize([800,800]),
        transforms.ToTensor()
    ])
)

# print(train_dataset)
# print(test_dataset)

# Hyper parameters
num_epochs = 5
num_classes = 2
batch_size = 32
learning_rate = 0.00001
#checkpoint_interval = 1
# Data loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

model = models.vgg16_bn(num_classes=num_classes).to(device)

# model = VGG(num_classes=num_classes).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

tb = SummaryWriter('logs')


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    tb.add_scalar('VGG16TrainLoss', loss.item(), epoch + 1)
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}
    path_checkpoint = "./models/checkpoint_{}_epoch.pth".format(epoch)
    torch.save(checkpoint, path_checkpoint)

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        tb.add_scalar('testAccuracy', 100 * correct / total, total)

    print('Test Accuracy of the model on the 6626 test images: {} %'.format(100 * correct / total))
tb.close()
# Save the model checkpoint
# torch.save(model.state_dict(), './models/model.pth')
