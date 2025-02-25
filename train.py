from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop
import numpy as np
from torch import nn, optim
import torch
from torchvision.models import mobilenet_v3_large
import argparse
from IE import Face_Dataset
from tqdm import tqdm
from tensorboardX import SummaryWriter

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义训练函数
img_transforms = Compose(
    [ToTensor(), Resize((512, 512)), RandomCrop(500), Resize((512, 512))]
)

# 定义学习率调整函数
def warmup_learning_rate(optimizer, iteration):
    lr_ini = 0.0001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_ini + (args.initial_lr - lr_ini) * iteration / 100

# 定义余弦退火学习率调整函数
def cosine_deacy(optimizer, lr_base, global_epoch, warmup_epoch, total_epochs):
    if global_epoch < warmup_epoch:
        lr = 0.5 * lr_base * (1 + np.cos(np.pi * global_epoch / warmup_epoch))
    else:
        lr = (
            0.5
            * lr_base
            * (
                1
                + np.cos(
                    np.pi
                    * (global_epoch - warmup_epoch)
                    / float(total_epochs - warmup_epoch)
                )
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# 定义训练函数
def train(args):
    # 定义网络，使用MobileNetV3
    net = nn.Sequential(mobilenet_v3_large(), nn.Linear(1000, 100), nn.Linear(100, 7))
    model = net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter("./log")
    # 训练
    for epoch in range(args.epochs):
        # 加载数据集
        face_data_orign = Face_Dataset(args.train_datasets, img_transforms)
        face_data_crop = Face_Dataset(args.train_datasets, img_transforms)
        face_data_equ = Face_Dataset(
            args.train_datasets, imgs_transform=img_transforms, equalize=True
        )
        face_data_con = Face_Dataset(
            args.train_datasets, imgs_transform=img_transforms, contrast_enhanced=True
        )
        face_data = ConcatDataset(
            [face_data_orign, face_data_crop, face_data_equ, face_data_con]
        )
        face_dataloader = DataLoader(
            face_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        # 展示进度
        print("#" + "_" * 40 + "#")
        for img, emotion in tqdm(face_dataloader):
            img = img.type(torch.FloatTensor)
            img = img.to(device)
            emotion = emotion.to(device)
            out = model(img)
            loss = criterion(out, emotion.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch == args.warmup_epoch:
                lr_base = optimizer.state_dict()["param_groups"][0]["lr"]
            if epoch >= args.warmup_epoch:
                cosine_deacy(optimizer, lr_base, epoch, args.warmup_epoch, args.epochs)
        writer.add_scalar("train_loss", loss / args.batch_size, global_step=epoch)
        # 打印训练信息
        print(
            "epoch:{0},train_loss:{1},learning_rate:{2}".format(
                epoch + 1,
                round(loss.item() / args.batch_size, 6),
                round(optimizer.state_dict()["param_groups"][0]["lr"], 6),
            )
        )
    # 保存模型
    torch.save(model, "{0}EMC{1}.pth".format(args.weights, epoch + 1))

# 定义参数解析函数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--initial_lr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup_epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=18, type=int)
    parser.add_argument("--weights", default="./weights/", type=str)
    parser.add_argument(
        "--train_datasets", default="./data/KDEF_order_train/", type=str
    )
    args = parser.parse_args()
    return args

# 主函数
if __name__ == "__main__":
    args = parse_args()
    args.epochs = int(input("Please input the number of epochs: "))
    train(args)
