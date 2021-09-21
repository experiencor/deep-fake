import os
import torch.utils.data
import torch
import random
import cv2
import albumentations as A
import logging
import numpy as np
import torch.nn.functional as F
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
)
from torchvision.transforms import (
    Compose,
    Lambda,
)


train_aug = A.Compose([
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    A.GaussNoise(p=0.1),
    A.GaussianBlur(blur_limit=3, p=0.05),
    A.HorizontalFlip(),
    A.OneOf([
            A.RandomBrightnessContrast(), 
            A.FancyPCA(), 
            A.HueSaturationValue()
        ], 
        p=0.7
    ),
    A.ToGray(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.1, 
        scale_limit=0.2, 
        rotate_limit=10, 
        border_mode=cv2.BORDER_CONSTANT, 
        p=0.5
    )
])

transform = Compose(
    [
    ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ]
        ),
        ),
    ]
)

def log(*args):
    print_mess = " ".join([str(arg) for arg in args])
    logging.warning(print_mess + "\n\n")

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def calc_prob(logits):
    return F.softmax(logits, dim=1)[:,1]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_num_crop_workers(each_worker=2.6):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    free = info.free/1e9
    return int(np.floor((free / each_worker)))
    
