import os

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0))

test_dataset = torchvision.datasets.ImageFolder(
    root='D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/test_data',
    transform=transforms.Compose([
        transforms.Resize([800,800]),
        transforms.ToTensor()
    ])
)

# Hyper parameters
num_classes = 2
batch_size = 8
# Data loader

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# tb = SummaryWriter('logs')



list_accuracy = []
for s in os.listdir('./merge_models/'):
    i = 0
    model1 = AlexNet(num_classes).to(device)
    checkpoint = torch.load('./models/checkpoint_2_epoch.pth')
    model1.load_state_dict(checkpoint['model_state_dict'])

    model2 = AlexNet(num_classes).to(device)
    checkpoint = torch.load('./origin_models/checkpoint_11_epoch.pth')
    model2.load_state_dict(checkpoint['model_state_dict'])

    MM = MergeModel(model1, model2).to(device)
    checkpoint = torch.load(os.path.join('./merge_models/',s))
    MM.load_state_dict(checkpoint['model_state_dict'])
    print('start_test:{}'.format(s))
    list_true = []
    list_pred = []
    # Test the model
    MM.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            list_true.extend(labels.tolist())
            outputs = MM(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            pred = predicted.tolist()
            list_pred.extend(pred)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        list_accuracy.append((s,accuracy))
        print('{} Test Accuracy {} %'.format(s, accuracy))

    conf_matrix = sklearn.metrics.confusion_matrix(list_true, list_pred)
    df_cm = pd.DataFrame(conf_matrix,
                         index=['non_Melanoma ', 'Melanoma'],
                         columns=['non_Melanoma', 'Melanoma'])
    f, ax = plt.subplots()
    sn.heatmap(df_cm, annot=True, cmap="BuPu", ax=ax, fmt="d")
    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig('./merge_plots/confusion_matrix{}.png'.format(i))
    plt.show()
    i = i+1
    # tb.close()
    # Save the model checkpoint
    # torch.save(epoch1_model.state_dict(), 'model.ckpt')
