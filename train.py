import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import pytorchvideo.models.resnet
import torch
import random
import time
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import argparse
import cv2
import albumentations as A
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import numpy as np
import wandb
from custom_labeled_dataset import CustomVideoDataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
)
from torchvision.transforms import (
    Compose,
    Lambda,
)

LEARNING_RATE       = 1e-4
SEED                = 100
EPOCH               = 2
ALPHA               = 4
FRAME_NUMBER        = 32
OCCLUDED_PERCENT    = 0.3
PREFETCH_FACTOR     = 2
NUM_WORKERS         = 4

# Dataset configuration
ROOT_PATH           = "/data/projects/deepfake"
BATCH_SIZE          = 10
FRAME_SIZE          = 224
CROP_SIZE           = 224
    
    
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
val_aug = A.Compose([
    A.HorizontalFlip(),
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

train_auc   = torchmetrics.AUROC(pos_label=1)
val_auc     = torchmetrics.AUROC(pos_label=1)
test_auc    = torchmetrics.AUROC(pos_label=1)
def seed_worker(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_top_prob(logits):
    return F.softmax(logits, dim=1)[:,1]
    

class CustomDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_version):
        super().__init__()
        self.train_dataset = CustomVideoDataset(
            f"{ROOT_PATH}/data/{data_version}/train.csv",
            frame_number=FRAME_NUMBER,
            frame_size=FRAME_SIZE,
            video_path_prefix="docker/faces/0/",
            augmentation=train_aug,
            transform=transform,
        )
        self.val_dataset = CustomVideoDataset(
            f"{ROOT_PATH}/data/{data_version}/dev.csv",
            frame_number=FRAME_NUMBER,
            frame_size=FRAME_SIZE,
            video_path_prefix="docker/faces/0/",
            augmentation=val_aug,
            transform=transform
        )
        self.test_dataset = CustomVideoDataset(
            f"{ROOT_PATH}/data/{data_version}/test.csv",
            frame_number=FRAME_NUMBER,
            frame_size=FRAME_SIZE,
            video_path_prefix="docker/faces/0/",
            transform=transform
        )   
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.train_dataset,
                timeout=1000000,
                batch_size=BATCH_SIZE,
                sampler=torch.utils.data.RandomSampler(self.train_dataset),
                num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                worker_init_fn=seed_worker
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.val_dataset,
                timeout=1000000,
                batch_size=BATCH_SIZE,
                sampler=torch.utils.data.SequentialSampler(self.val_dataset),
                num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                worker_init_fn=seed_worker
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.test_dataset,
                timeout=1000000,
                batch_size=BATCH_SIZE,
                sampler=torch.utils.data.SequentialSampler(self.test_dataset),
                num_workers=NUM_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                worker_init_fn=seed_worker
        )

    def get_trainset_size(self):
        return len(self.train_dataset)


class CustomClassificationLightningModule(LightningModule):
    def __init__(self, total_steps):
        super().__init__()
        self.model = pytorchvideo.models.resnet.create_resnet(
            input_channel=3, # RGB input from Kinetics
            model_depth=50, # For the tutorial let's just use a 50 layer network
            model_num_class=2, # Kinetics has 400 classes so we need out final head to align
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )
        self.best_auc = 0
        self.total_steps = total_steps
        self.automatic_optimization = False
    
    def log_all(self, metrics):
        wandb.log(metrics, step=self.global_step)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()        
        opt.zero_grad()
        
        lr = [group['lr'] for group in opt.param_groups][0]
        logits = self.model(batch["video"])
        loss = F.cross_entropy(logits, batch["label"])
        mean_loss = torch.mean(self.all_gather(loss))
        train_auc.update(get_top_prob(logits), batch["label"])

        self.manual_backward(loss)
        opt.step()
        
        if self.global_rank == 0:
            self.log_all({
                "train/loss": mean_loss,
                "lr": lr
            })
    
    def training_epoch_end(self, outs):
        auc = train_auc.compute()
        train_auc.reset()
        
        if self.global_rank == 0:
            self.log_all({
                "train/auc": auc
            })
            
    def validation_step(self, batch, batch_idx):            
        logits = self.model(batch["video"])
        loss = F.cross_entropy(logits, batch["label"])
        mean_loss = torch.mean(self.all_gather(loss))
        val_auc.update(get_top_prob(logits), batch["label"])
        
        if self.global_rank == 0:
            self.log_all({
               "val/loss": mean_loss
            })
    
    def validation_epoch_end(self, outs):
        auc = val_auc.compute()
        val_auc.reset()
        
        sch = self.lr_schedulers()
        sch.step(auc)
        
        if self.global_rank == 0:
            self.best_auc = max(auc, self.best_auc)
            self.log_all({
                "val/auc": auc,
                "best_auc": self.best_auc,
                "epoch": self.current_epoch
            })
        self.log("val/auc", auc, sync_dist=True)
    
    def test_step(self, batch, batch_idx):            
        logits = self.model(batch["video"])
        test_auc.update(get_top_prob(logits), batch["label"])
    
    def test_epoch_end(self, outs):
        auc = test_auc.compute()
        test_auc.reset()

        if self.global_rank == 0:
            self.log_all({
                "test/auc": auc
            })
        self.log("test/auc", auc, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max",
            patience=3,
            cooldown=1,
            factor=0.3,
            verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/auc"}
    

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(SEED)
    num_gpu = torch.cuda.device_count()
    
    config = {
        "data_version":     args.data_version,
        "code_version":     args.code_version,
        "pret_version":     args.pret_version,
        "num_gpu":          num_gpu,
        "batch_size":       BATCH_SIZE * num_gpu,
        "image_size":       CROP_SIZE,
        "max_lr":           LEARNING_RATE
    }
    run = wandb.init(project="deepfake", config=config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/auc",
        dirpath=f"{ROOT_PATH}/output/{run.id}",
        filename="model",
        save_top_k=1,
        mode="max",
    )
    
    data_module = CustomDataModule(args.data_version)
    total_steps = EPOCH * int(np.ceil(data_module.get_trainset_size() / (num_gpu * BATCH_SIZE)))
    classification_module = CustomClassificationLightningModule(
        total_steps=total_steps
    )
    pretrain = torch.load(f"{ROOT_PATH}/pretrain/{args.pret_version}/model.ckpt")
    if "state_dict" in pretrain:
        pretrain = pretrain["state_dict"]
    missing_keys, unexpected_keys = classification_module.load_state_dict(pretrain)
    print("missing_keys   :\t", missing_keys)
    print("unexpected_keys:\t", unexpected_keys)
    
    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=0,
        max_epochs=EPOCH,
        gpus=num_gpu,
        callbacks=[checkpoint_callback],
        accelerator="ddp", 
        precision=16,
    )
    
    trainer.fit(classification_module, data_module)
    time.sleep(10)
    best_path = os.path.join(ROOT_PATH, "output/best_model/model.ckpt")
    state_dict = torch.load(best_path)["state_dict"]
    classification_module.load_state_dict(state_dict)
    set_seed(SEED)
    trainer.test(classification_module, [data_module.test_dataloader()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data-version', type=str)
    parser.add_argument('--code-version', type=str)
    parser.add_argument('--pret-version', type=str)
    args = parser.parse_args()
    main(args)