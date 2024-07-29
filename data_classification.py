# 图片分类成 患病和没患病的 ，以文件夹名作为标签
import os
import shutil
import pandas as pd


def imgClassification(imgDir, digital_data_dir, digital_data_name):
    train_digital_data = pd.read_csv(os.path.join(digital_data_dir, digital_data_name), encoding='gbk', header=0)
    Melanoma_path = os.path.join(imgDir, 'Melanoma')
    Non_Melanoma_path = os.path.join(imgDir, 'Non_Melanoma')
    if os.path.exists(Melanoma_path) == False:
        os.makedirs(Melanoma_path)
    if os.path.exists(Non_Melanoma_path) == False:
        os.makedirs(Non_Melanoma_path)
    for i in range(len(train_digital_data)):
        if train_digital_data['target'][i] == 0:
            imgName = train_digital_data['image_name'][i] + '.jpg'
            shutil.move(os.path.join(imgDir, imgName), Non_Melanoma_path)
        else:
            imgName = train_digital_data['image_name'][i] + '.jpg'
            shutil.move(os.path.join(imgDir, imgName), Melanoma_path)
    return Melanoma_path, Non_Melanoma_path


if __name__ == '__main__':
    imgDir = ''  # 你图片数据的路径，不在项目文件及下使用绝对路径  注意在python里路径分隔符是 /，直接从资源管理器复制过来的需要吧\改成/
    digital_data_dir = ''  # 你CSV数据的路径，不在项目文件及下使用绝对路径  注意在python里路径分隔符是 /，直接从资源管理器复制过来的需要吧\改成/
    digital_data_name = ''  # 你CSV数据的名称
    Melanoma_path,Non_Melanoma_path = imgClassification(imgDir,digital_data_dir,digital_data_name)
    print(os.listdir(Melanoma_path))
    print(os.listdir(Non_Melanoma_path))

    #Melanoma_path应该有584张图片