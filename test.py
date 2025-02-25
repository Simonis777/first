import numpy as np  
import pandas as pd  
import torch  
import torch.nn as nn  
import torchvision as tv  
from torchvision.models import mobilenet_v3_large
import torchvision.transforms as transforms  
import cv2 as cv  
import seaborn as sns  
from sklearn.metrics import confusion_matrix  
from torch.autograd import no_grad  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
import matplotlib.pyplot as plt  
from unicodedata import normalize  
from IE import Face_Dataset, Face_Test_Dataset 
from operator import index
import argparse 

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义数据增强
img_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((512, 512))]
)
# 定义情绪类别
emotion_list = ["afraid", "angry", "disgust", "happy", "netural", "sad", "suprise"]
emotion_label = {"AF": 0, "AN": 1, "DI": 2, "HA": 3, "NE": 4, "SA": 5, "SU": 6}

# 测试模型
def test(args):
    model = torch.load(args.weights)
    model = model.to(device)
    # 加载数据集
    test_datasets = Face_Test_Dataset(args.test_datasets, img_transforms)
    test_dataloader = DataLoader(
        test_datasets,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False,
    )
    predict_correct_num = 0
    label_num = 0
    wrong_num = 0
    target_predict = []
    target_label = []
    class_num = [str(i) for i in range(7)]
    bar = tqdm(test_dataloader)
    model.eval()
    # 测试,计算准确率
    with no_grad():
        for img, emotion in tqdm(bar):
            img = img.type(torch.FloatTensor)
            img = img.to(device)
            emotion_label = emotion.to(device)
            emotion_predict = model(img)
            emotion_predict = emotion_predict.squeeze()
            emotion_predict = emotion_predict.cpu().numpy()
            emotion_predict = np.where(emotion_predict == np.max(emotion_predict))
            emotion_label = emotion_label.cpu().numpy()
            label_num += 1
            # 判断预测是否正确
            if emotion_predict[0] == emotion_label[0]:
                # 预测正确
                predict_correct_num += 1
            else:
                # 预测错误
                wrong_num += 1
                save_wrong_path = (
                    args.predict_wrong
                    + "origin"
                    + "_"
                    + emotion_list[int(emotion_label[0])]
                    + "_"
                    + "predict"
                    + "_"
                    + emotion_list[int(emotion_predict[0])]
                    + "__"
                    + str(wrong_num)
                    + ".jpg"
                )
                # 转换成原本的图片
                img = img.cpu().detach().numpy()
                img = img.squeeze()
                img = img.transpose(1, 2, 0)
                img = img * 255
                img = img.astype(np.uint8)
                # 保存错误图片
                cv.imwrite(save_wrong_path, img)
            # 记录预测结果
            target_label.append(str(int(emotion_label[0])))
            target_predict.append(str(int(emotion_predict[0])))
            # 更新进度条
            bar.set_description(
                "accuracy_rate of emotion classification is %f"
                % (predict_correct_num / label_num)
            )

    # 计算准确率
    target_label = np.array(target_label)
    target_predict = np.array(target_predict)
    # 计算混淆矩阵
    matrix = confusion_matrix(target_label, target_predict)
    # 画混淆矩阵
    dataframe = pd.DataFrame(matrix, index=emotion_list, columns=emotion_list)
    C_M = sns.heatmap(dataframe, annot=True, cbar=None, cmap="BuPu")
    plt.title("Accuracy")
    plt.ylabel("emotion_label")
    plt.xlabel("emotion_predict")
    plt.show()
    C_M = C_M.get_figure()
    C_M.savefig("Confusion_Matrix_Emotion_Classification.jpg", dpi=500)

# 判断单个图片的情绪，返回百分比
def judge(pic_path):
    args = parse_args()
    return judge_ture(pic_path,args)

# 判断单个图片的情绪
def judge_ture(pic_path,args):
    # 加载模型
    model = torch.load(args.weights)
    model = model.to(device)
    img = cv.imread(pic_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    model.eval()

    with no_grad():
        emotion_predict = model(img)
        emotion_predict = emotion_predict.squeeze()
        emotion_predict = emotion_predict.cpu().numpy()
        emotion_predict = np.exp(emotion_predict) / np.sum(np.exp(emotion_predict))
        emotion_predict = emotion_predict * 100
        emotion_predict = np.round(emotion_predict, 2)
        emotion_predict = emotion_predict.tolist()
    # 返回预测结果
    return emotion_predict

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--test_datasets", default="./data/KDEF_ORDER_test/", type=str)
    parser.add_argument(
        "--weights", default="./weights/EMC30.pth", type=str
    )
    parser.add_argument(
        "--predict_wrong", default="./data/predict_wrong/", type=str
    )
    args = parser.parse_args()
    return args

# 主函数
if __name__ == "__main__":
    choice = input("1.测试模型的准确率和混淆矩阵\n2.判断单个图片的情绪\n")
    if choice == "1":
        args = parse_args()
        test(args)
    elif choice == "2":
        pic_path = r"E:\MyFirstAI\KDEF\AF01\AF01DIHR.JPG"
        percent = judge(pic_path)
        print("预测的百分比为:", percent)
