from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np
import torch

# 定义情绪类别,AF:afraid,AN:angry,DI:disgust,HA:happy,NE:neutral,SA:sad,SU:surprise
emotion_label = {"AF": 0, "AN": 1, "DI": 2, "HA": 3, "NE": 4, "SA": 5, "SU": 6}

# 生成one-hot编码
def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

# 生成gamma变换
def gamma(image):
    image = image / 255.0
    gamma = 0.4
    image = np.power(image, gamma)
    return image

# 生成对比度增强
def CLHE(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  #createCLAHE函数用于自适应直方图均衡化
    image = clahe.apply(image)
    return image

# 训练数据集
class Face_Dataset(Dataset):
    # 初始化
    def __init__(
        self, img_dir=None, imgs_transform=None, equalize=False, contrast_enhanced=False
    ):
        self.img_dir = img_dir
        self.transform = imgs_transform
        self.filelist = os.listdir(self.img_dir)
        self.equalize = equalize
        self.contrast = contrast_enhanced
        self.gORm = 0

    def __len__(self):
        return len(self.filelist)

    # 获取数据
    def __getitem__(self, index):
        img_name = self.img_dir + self.filelist[index]
        temp_img = cv.imread(img_name)
        # 直方图均衡
        if self.equalize == True:
            b, g, r = cv.split(temp_img)

            b1 = cv.equalizeHist(b)
            g1 = cv.equalizeHist(g)
            r1 = cv.equalizeHist(r)
            temp_img = cv.merge([b1, g1, r1])
        # 对比度增强
        if self.contrast == True:
            if self.gORm == 0:
                b2, g2, r2 = cv.split(temp_img)
                b2 = gamma(b2)
                g2 = gamma(g2)
                r2 = gamma(r2)
                temp_img = cv.merge([b2, g2, r2])

                self.gORm = 1
            else:
                b3, g3, r3 = cv.split(temp_img)
                b3 = CLHE(b3)
                g3 = CLHE(g3)
                r3 = CLHE(r3)
                temp_img = cv.merge([b3, g3, r3])

                self.gORm = 0
        # 获取标签
        label_index = self.filelist[index].split(".")[0][:2]

        emotion = emotion_label[label_index]
        # 转换图片
        if self.transform is not None:
            gray_pic = self.transform(temp_img)

        emotion = torch.LongTensor([emotion])
        return gray_pic, emotion

# 测试数据集
class Face_Test_Dataset(Dataset):
    def __init__(self, img_dir=None, imgs_transform=None):
        self.img_dir = img_dir
        self.transform = imgs_transform
        self.filelist = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_name = self.img_dir + self.filelist[index]
        temp_img = cv.imread(img_name)
        label_index = self.filelist[index].split(".")[0][:2]
        emotion = emotion_label[label_index]

        if self.transform is not None:
            gray_pic = self.transform(temp_img)

        emotion = torch.LongTensor([emotion])
        return gray_pic, emotion
