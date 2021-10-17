import os
import pytorch_lightning
import torch.utils.data
import torch
import pandas as pd
import albumentations as A
from utils import train_aug, transform, set_seed
from dataset import Dataset


class DataLoader(pytorch_lightning.LightningDataModule):
    def __init__(self, 
        data_version, 
        root_path, 
        frame_number, 
        frame_size, 
        batch_size, 
        frame_num,
        freq_num, 
        audio_len, 
        num_eval_iters
    ):
        super().__init__()
        self.data = pd.read_csv(f"{root_path}/data/{data_version}.csv")
        self.train_data = self.data[self.data.split == "train"].copy()
        dev_data        = self.data[self.data.split == "dev"].copy()
        test_data       = self.data[self.data.split == "test"].copy()

        self.epoch = 0
        self.batch_size = batch_size
        self.num_eval_iters = num_eval_iters

        self.train_dataset = Dataset(
            data_frame=self.train_data,
            epoch=self.epoch,
            frame_number=frame_number,
            transform=transform,
            augmentation=train_aug,
            frame_size=frame_size,
            frame_num=frame_num,
            freq_num=freq_num,
            audio_len=audio_len,
        )
        self.val_dataset = Dataset(
            data_frame=dev_data,
            epoch=self.epoch,
            frame_number=frame_number,
            transform=transform,
            augmentation=train_aug,
            frame_size=frame_size,
            frame_num=frame_num,
            freq_num=freq_num,
            audio_len=audio_len,
        )
        self.test_dataset = Dataset(
            data_frame=test_data,
            epoch=self.epoch,
            frame_number=frame_number,
            transform=transform,
            frame_size=frame_size,
            frame_num=frame_num,
            freq_num=freq_num,
            audio_len=audio_len,
        )

    def train_dataloader(self):
        self.train_dataset._epoch = self.epoch % self.num_eval_iters
        self.epoch += 1
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.RandomSampler(self.train_dataset),
            num_workers=8,
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=set_seed
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(self.val_dataset),
            num_workers=8,
            prefetch_factor=3,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(self.test_dataset),
            num_workers=4,
            prefetch_factor=2,
        )

    def get_trainset_size(self):
        return len(self.train_data)