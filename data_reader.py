from torch.utils.data.dataset import Dataset
import os
from PIL import Image

class MyDataSet(Dataset):
    def __init__(self, root_path, label_dir):
        self.root_path = root_path
        self.label_dir = label_dir
        self.path = os.path.join(self.root_path, self.label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        image_name = self.image_path[index]
        image_item_path = os.path.join(self.root_path, self.label_dir, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir
        return image, label

    def __len__(self):
        return len(self.image_path)


root_path = 'D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train'
Melanoma_dir = '1'
Non_Melanoma_dir = '0'
Melanoma_dataset = MyDataSet(root_path,Melanoma_dir)
Non_Melanoma_dataset = MyDataSet(root_path,Non_Melanoma_dir)
train_dataset = Melanoma_dataset + Non_Melanoma_dataset
img, label = train_dataset[583]
print(label)