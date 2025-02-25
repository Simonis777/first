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
print(torch.__version__)
print(torch.version.cuda)
