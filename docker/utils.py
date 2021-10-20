import os
import torch.utils.data
import torch
import random
import cv2
import albumentations as A
import logging
import numpy as np
import json
from pynvml import *
import torch.nn.functional as F
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
)
import cv2
import math
import PIL
import albumentations as A
import torchvision
import augly.image as imaugs

config = json.load(open("config.json"))
image_size = config["video_size"]


def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

    return img_ycbcr

def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    # to [0, 1]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    # to [0, 1]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_bgr

def overlay_text(image, p=0.5, **kwargs):
    if np.random.rand() < p:
        x = np.random.randint(image_size//6-20, image_size//6+20)
        y = np.random.randint(image_size//2-20, image_size//2+20)
        s = np.random.randint(1, 4)
        image = cv2.putText(
            img=np.copy(image), 
            text="hello.com", 
            org=(x,y),
            fontFace=1, 
            fontScale=s, 
            color=(0,0,255), 
            thickness=2
        )
    return np.array(image)

def overlay_emoji(image, p=0.5, **kwargs):
    if np.random.rand() < p:
        x = np.random.rand()
        y = np.random.rand()
        image = imaugs.overlay_emoji(
            PIL.Image.fromarray(image[:,:,::-1]),
            x_pos=x,
            y_pos=y
        )
        image = np.array(image)[:,:,:3][:,:,::-1]
    return image

def gaussian_noise_color(img, param=0.005, p=0.5, **kwargs):
    if np.random.rand() < p:
        ycbcr = bgr2ycbcr(img) / 255
        size_a = ycbcr.shape
        b = (ycbcr + math.sqrt(param) *
             np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
        b = ycbcr2bgr(b)
        img = np.clip(b, 0, 255).astype(np.uint8)

    return img

train_aug = A.Compose([
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    A.GaussianBlur(blur_limit=(3,3), p=0.05),
    A.Lambda(gaussian_noise_color),
    A.HorizontalFlip(),
    A.ColorJitter(
        brightness=0.5, 
        contrast=0.5, 
        saturation=0.5, 
        hue=0.5, 
        p=0.5
    ),
    A.OneOf([
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
    ),
    A.CoarseDropout(
        min_holes=5,
        max_holes=8, 
        min_height=16,
        max_height=24, 
        min_width=16,
        max_width=24, 
        fill_value=0, 
        p=0.5
    ),
    A.Lambda(overlay_text),
    #A.Lambda(overlay_emoji),
    A.RandomSizedCrop(
        [image_size*3//4, image_size], 
        image_size, 
        image_size, 
        w2h_ratio=1.0, 
        interpolation=1, 
        p=0.2
    ),
])

class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(config["video_len"]),
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            PackPathway()
        ]
    ),
)

audio_transform = Compose(
    [
        torchvision.transforms.Normalize((0.45, 0.45), (0.225, 0.225))
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

def compute_num_crop_workers(each_worker=8):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    free = info.free/1e9
    return int(np.ceil((free / each_worker)))

def predict(trainer, model, dataloader):
    logits = trainer.predict(model, dataloader)
    logits = torch.cat(logits)
    probs  = calc_prob(logits).cpu().detach().numpy()
    predictions = [{"filename": example["filename"], "prob": prob} \
        for prob, example in zip(probs, dataloader.dataset)]
    return predictions
    
