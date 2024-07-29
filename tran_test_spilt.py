import os
import shutil

train_path_0 = './ISIC_2020_Training_JPEG/train/train_data/0/'
train_path_1 = './ISIC_2020_Training_JPEG/train/train_data/1/'
test_path_0 = './ISIC_2020_Training_JPEG/train/test_data/0/'
test_path_1 = './ISIC_2020_Training_JPEG/train/test_data/1/'

if os.path.exists(train_path_0) == False:
    os.makedirs(train_path_0)
if os.path.exists(train_path_1) == False:
    os.makedirs(train_path_1)
if os.path.exists(test_path_0) == False:
    os.makedirs(test_path_0)
if os.path.exists(test_path_1) == False:
    os.makedirs(test_path_1)

root_train_0 = 'D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/0'
root_train_1 = 'D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/1'

train_size1 = int(0.8 * len(os.listdir(root_train_0)))
train_size2 = int(0.8 * len(os.listdir(root_train_1)))
for i in range(train_size1):
    shutil.copy(os.path.join(root_train_0, os.listdir(root_train_0)[i]), train_path_0)
for i in range(train_size2):
    shutil.copy(os.path.join(root_train_1, os.listdir(root_train_1)[i]), train_path_1)

for i in range(train_size1, len(os.listdir(root_train_0))):
    shutil.copy(os.path.join(root_train_0, os.listdir(root_train_0)[i]), test_path_0)
for i in range(train_size2, len(os.listdir(root_train_1))):
    shutil.copy(os.path.join(root_train_1, os.listdir(root_train_1)[i]), test_path_1)