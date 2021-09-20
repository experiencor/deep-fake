import os
import pytorch_lightning
import torch.nn as nn
import pytorchvideo.data
import torch.utils.data
import pytorchvideo.models.resnet
import torch
import time
import random
from utils import log
import pandas as pd
import logging
import torch.nn.functional as F
from argparse import ArgumentParser
from pytorch_lightning.core.lightning import LightningModule
import numpy as np
from custom_labeled_dataset import CustomVideoDataset
from pynvml import *
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
)
from torchvision.transforms import (
    Compose,
    Lambda,
)

DEVICE              = "cpu"
NUM_ITERATIONS      = 1

EVAL_BATCH_SIZE     = 1
EVAL_NUM_WORKERS    = 1
PREFETCH_FACTOR     = 1

CROP_BATCH_SIZE     = 4
CROP_NUM_WORKERS    = 2
CROP_THRESHOLD      = 0.975

FRAME_NUMBER        = 4
FRAME_SIZE          = 224

SEED                = 100
ALPHA               = 4
ROOT_PATH           = "."


nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
multiplier = int(np.ceil((info.free/1e9) / 5))
logging.warning(f"multiplier: {multiplier}")

CROP_NUM_WORKERS    = CROP_NUM_WORKERS * max(1, multiplier)
EVAL_BATCH_SIZE     = EVAL_BATCH_SIZE  * max(1, multiplier)


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


def get_positive_prediction(logits):
    return F.softmax(logits, dim=1)[:,1]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class CustomDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, input_dir):
        super().__init__()
        self.test_dataset = CustomVideoDataset(
            os.path.join(ROOT_PATH, "test.csv"),
            frame_number=FRAME_NUMBER,
            frame_size=FRAME_SIZE,
            video_path_prefix="faces/0/",
            transform=transform
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            timeout=1000000,
            batch_size=EVAL_BATCH_SIZE,
            sampler=torch.utils.data.SequentialSampler(self.test_dataset),
            num_workers=EVAL_NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR
        )
    

class CustomClassificationLightningModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = pytorchvideo.models.resnet.create_resnet(
            input_channel=3, # RGB input from Kinetics
            model_depth=50, # For the tutorial let's just use a 50 layer network
            model_num_class=2, # Kinetics has 400 classes so we need out final head to align
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch["video"])


def main(input_dir, output_file):
    if not os.path.exists("/data/face"):
        os.mkdir("/data/face")
        
    set_seed(SEED)
    num_gpu = 0#torch.cuda.device_count()

    # read the video files and make the test file
    logging.warning("cropping videos")
    tik = time.time()
    test_videos = []
    with open("test.csv", "w") as f:
        for video in os.listdir(input_dir):
            if ".mp4" in video:
                f.write(f"{video} 1\n")
                test_videos += [video]
                
    # crop the faces from the videos    
    os.system(f"python crop.py --workers {CROP_NUM_WORKERS} --input {input_dir} --output faces")
    avg_crop_time = (time.time() - tik) / len(test_videos)
    logging.warning(f'Face cropping completed! Average crop time: {avg_crop_time} for {len(test_videos)} videos.')
    time.sleep(15)

    face_count = 0
    for face in os.listdir("faces/0/"):
        if ".npy" in face:
            face_count += 1
    if face_count != len(test_videos):
        return
    
    # create the model and load the weights
    classification_module = CustomClassificationLightningModule()
    
    if num_gpu == 0:
        pretrain = torch.load("model.ckpt", map_location=torch.device('cpu'))
    else:
        pretrain = torch.load("model.ckpt")
        
    if "state_dict" in pretrain:
        pretrain = pretrain["state_dict"]
    missing_keys, unexpected_keys = classification_module.load_state_dict(pretrain)
    log("missing_keys   :\t", missing_keys)
    log("unexpected_keys:\t", unexpected_keys)
    
    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=0,
        gpus=num_gpu,
        accelerator="ddp"
    )
    
    # perform predictions
    data_module = CustomDataModule(input_dir)
    final_results = []
    for _ in range(NUM_ITERATIONS):
        logits = trainer.predict(classification_module, data_module.test_dataloader())
        logits = torch.cat(logits)
        probs = get_positive_prediction(logits)
        final_results += [probs]
    final_results = torch.mean(torch.stack(final_results), dim=0).cpu().detach().numpy()
    
    # create output and write the ouput as csv
    output_df = pd.DataFrame({"filename": test_videos, "probability": final_results})
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-input",  type=str, required=True, help="Input directory of test videos")
    parser.add_argument("-output", type=str, required=True, help="Output directory with filename e.g. /data/output/submission.csv")
    args = parser.parse_args()

    main(input_dir=args.input, output_file=args.output)