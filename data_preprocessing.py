import os
import cv2 as cv
import numpy as np
import random

path = "./KDEF/"

# 数据集路径
save_path = "./data/KDEF_order/"
save_path_train = "./data/KDEF_order_train/"
save_path_test = "./data/KDEF_order_test/"

# 确认路径
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


check_dir(save_path)
check_dir(save_path_train)
check_dir(save_path_test)

# 保存每个类别的图片数量
abr_dict = {"AF": 0, "AN": 0, "DI": 0, "HA": 0, "NE": 0, "SA": 0, "SU": 0}


# 遍历文件夹
files = os.listdir(path)
for file_dir in files:

    for file in os.listdir(path + file_dir):
        temp = file.split(".")
        # 获取表情信息
        abr = temp[0][4:6]

        if abr in abr_dict.keys():
            temp_img = cv.imread(path + file_dir + '/' + file)
            # 判断图片是否为空
            if np.mean(temp_img) > 30:
                abr_dict[abr] += 1
                file_name = save_path + abr + str(abr_dict[abr]) + ".jpg"


                temp_img = cv.resize(temp_img, (762, 762))
                cv.imwrite(file_name, temp_img)
            # 图片为空则记录
            else:
                print(path + file + "/" + file)

                with open("./data/error_pic.txt", "a") as f:
                    f.write(path + file + "/" + file + "\n")

# 划分数据集
files = os.listdir(save_path)
length = len(files)
rand_seed = random.sample(range(0, length), length)
rand_seed_test = rand_seed[4000:]
rand_seed_train = rand_seed[:4000]


for i in rand_seed_train:
    train_read_path = save_path + files[i]
    train_save_path = save_path_train + files[i]
    img = cv.imread(train_read_path)
    cv.imwrite(train_save_path, img)

for i in rand_seed_test:
    test_read_path = save_path + files[i]
    test_save_path = save_path_test + files[i]
    img = cv.imread(test_read_path)
    cv.imwrite(test_save_path, img)

print("数据集划分完成")
