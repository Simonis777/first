import torch
import torchvision
from torch.utils.data import DataLoadewhole.pyr, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import DatasetFolder
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm
import argparse

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据增强
img_crop_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomCrop(500),
        transforms.Resize((512, 512)),
    ]
)

# 数据预处理
img_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((512, 512))]
)

# ----------------------------------------#
# 学习率调整函数
# ----------------------------------------#
def warmup_learning_rate(optimizer, iteration, args):
    lr_ini = 0.0001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_ini + (args.initial_lr - lr_ini) * iteration / 100


def cosin_decay(optimizer, lr_base, current_epoch, warmup_epoch, global_epoch):
    lr_new = (
        0.5
        * lr_base
        * (
            1
            + torch.cos(
                torch.tensor(
                    (current_epoch - warmup_epoch)
                    / float(global_epoch - warmup_epoch)
                    * 3.141592653589793
                )
            )
        )
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_new


# 训练函数
def train(args):
    net = nn.Sequential(mobilenet_v3_small(), nn.Linear(1000, 100), nn.Linear(100, 7))
    model = torch.nn.DataParallel(net, device_ids=[0, 1])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter("./log")

    # 训练
    for epoch in range(args.epochs):
        face_data_orign = DatasetFolder(args.train_datasets, loader=torchvision.datasets.folder.default_loader, extensions=("jpg", "jpeg", "png"), transform=img_transforms)
        face_data_crop = DatasetFolder(args.train_datasets, loader=torchvision.datasets.folder.default_loader, extensions=("jpg", "jpeg", "png"), transform=img_crop_transforms)
        face_data_equ = DatasetFolder(args.train_datasets, loader=torchvision.datasets.folder.default_loader, extensions=("jpg", "jpeg", "png"), transform=img_crop_transforms, equalize=True)
        face_data_con = DatasetFolder(args.train_datasets, loader=torchvision.datasets.folder.default_loader, extensions=("jpg", "jpeg", "png"), transform=img_crop_transforms, contrast_enhanced=True)
        face_data = ConcatDataset([face_data_orign, face_data_crop, face_data_equ, face_data_con])
        face_dataloader = DataLoader(
            face_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        print("#" + "_" * 40 + "#")
        for img, emotion in tqdm(face_dataloader):
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
                cosin_decay(optimizer, lr_base, epoch, args.warmup_epoch, args.epochs)
        writer.add_scalar("train_loss", loss / args.batch_size, global_step=epoch)
        print(
            "epoch:{0},train_loss:{1},learning_rate:{2}".format(
                epoch + 1,
                round(loss.item() / args.batch_size, 6),
                round(optimizer.state_dict()["param_groups"][0]["lr"], 6),
            )
        )
    torch.save(model.state_dict(), "{0}EMC{1}.pth".format(args.weights, epoch + 1))


# 解析参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--initial_lr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup_epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=18, type=int)
    parser.add_argument("--weights", default="./weights/", type=str)
    parser.add_argument(
        "--train_datasets", default="./data/KDEF_ORDER_TRAIN/", type=str
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)