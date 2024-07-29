from PIL import Image, ImageChops
from PIL import ImageEnhance
import os
import numpy as np
#移动类
def move_x(root_path,img_name,x): #平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offset = ImageChops.offset(img,x,0)
    sign = 'offsetX'
    return offset,sign
def move_y(root_path,img_name,y): #平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offset = ImageChops.offset(img,0,y)
    sign = 'offsetY'
    return offset,sign
def flip_left_right(root_path,img_name):   #左右翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    sign = "flipLeft"
#     filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img,sign
def flip_top_bottom(root_path,img_name):   #上下翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    sign = "flipTop"
#     filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img,sign
def rotation(root_path, img_name,angle):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(angle) #旋转角度
    sign = 'rotation'
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img,sign


#色彩类
def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    sign = 'randomColor'
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor),sign # 调整图像锐度
def contrastEnhancement(root_path,img_name):#对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 3
    image_contrasted = enh_con.enhance(contrast)
    sign = 'contrastEnhancement'
    return image_contrasted,sign
def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.4
    image_brightened = enh_bri.enhance(brightness)
    sign = 'brightnessEnhancement'
    return image_brightened,sign
def colorEnhancement(root_path,img_name):#颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 2
    image_colored = enh_col.enhance(color)
    sign = 'colorEnhancement'
    return image_colored,sign



if __name__ == '__main__':
    # move_x  move_y  flip_left_right  flip_top_bottom  rotation         randomColor contrastEnhancement brightnessEnhancement colorEnhancement
    # 1色彩+5移动  2的6次方
    imageDir1 = 'D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/cla2/1/'  # 要增强的文件夹
    for name in os.listdir(imageDir1):
        saveImage, sign = move_x(imageDir1, name, 400)
        saveName = name.replace('.jpg', '') + '_' + sign + ".jpg"
        saveImage.save(os.path.join(imageDir1, saveName))

    for name in os.listdir(imageDir1):
        saveImage, sign = rotation(imageDir1, name, 25)
        saveName = name.replace('.jpg', '') + '_' + sign + ".jpg"
        saveImage.save(os.path.join(imageDir1, saveName))

    for name in os.listdir(imageDir1):
        saveImage, sign = flip_left_right(imageDir1, name)
        saveName = name.replace('.jpg', '') + '_' + sign + ".jpg"
        saveImage.save(os.path.join(imageDir1, saveName))

    for name in os.listdir(imageDir1):
        saveImage, sign = move_y(imageDir1, name, 400)
        saveName = name.replace('.jpg', '') + '_' + sign + ".jpg"
        saveImage.save(os.path.join(imageDir1, saveName))

    for name in os.listdir(imageDir1):
        saveImage, sign = flip_top_bottom(imageDir1, name)
        saveName = name.replace('.jpg', '') + '_' + sign + ".jpg"
        saveImage.save(os.path.join(imageDir1, saveName))

    for name in os.listdir(imageDir1):
        saveImage, sign = contrastEnhancement(imageDir1, name)
        saveName = name.replace('.jpg', '') + '_' + sign + ".jpg"
        saveImage.save(os.path.join(imageDir1, saveName))


#多余删减
    # image_dir = 'D:/Jupyter/基于深度学习的图像生成和图像识别/黑色素肿瘤分类/ISIC_2020_Training_JPEG/train/cla2/1'  # 增强的文件夹
    # ls = []
    # for s in os.listdir(image_dir):
    #     if 'contrastEnhancement' in s:
    #         if len(s) >= 60:
    #             ls.append(s)
    # # print(len(ls))
    # for i in range(3855):  # 数字填1文件夹图片-0文件夹图片数量
    #     os.remove(os.path.join(image_dir, ls[i]))
