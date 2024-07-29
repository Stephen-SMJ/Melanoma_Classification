import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from networks import AlexNet
import seaborn as sn
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
import os

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0))

test_dataset = torchvision.datasets.ImageFolder(
    root='D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/test2/',
    transform=transforms.Compose([
        transforms.Resize([800,800]),
        transforms.ToTensor()
    ])
)

# Hyper parameters
num_classes = 2
batch_size = 32
# Data loader

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

list_true = []
list_pred = []
list_accuracy=[]
tb = SummaryWriter('logs')
for s in os.listdir('./origin_models/'):
    i = 0
    model = AlexNet(num_classes).to(device)
    checkpoint = torch.load(os.path.join('./origin_models/',s))
    print('start_test:{}'.format(s))
    model.load_state_dict(checkpoint['model_state_dict'])
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
            pred = predicted.tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print('{} Test Accuracy {} %'.format(s,accuracy))
        list_accuracy.append(accuracy)
        tb.add_scalar('ds_accuarcy', accuracy, i + 1)
    i += i
    conf_matrix = sklearn.metrics.confusion_matrix(list_true, list_pred)
    df_cm = pd.DataFrame(conf_matrix,
                         index=['non_Melanoma ', 'Melanoma'],
                         columns=['non_Melanoma', 'Melanoma'])
    f, ax = plt.subplots()
    sn.heatmap(df_cm, annot=True, cmap="BuPu", ax=ax, fmt="d")
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig('./ds_test/confusion_matrix{}.png'.format(i))
tb.close()
print(max(list_accuracy),list_accuracy.index(max(list_accuracy)))
