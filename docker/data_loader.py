import os
import pytorch_lightning
import torch.utils.data
import torch
import pandas as pd
import albumentations as A
from utils import train_aug, transform, set_seed
from dataset import Dataset


class DataLoader(pytorch_lightning.LightningDataModule):
    def __init__(self, data_version, root_path, frame_number, frame_size, batch_size):
        super().__init__()
        self.video_path = f"{root_path}/data/{data_version}"

        splits = pd.read_csv("splits.csv")
        all_files = set(os.listdir(f"{self.video_path}/0"))
        splits = splits[splits.filename.isin(all_files)].copy()

        self.train_data = splits[splits.split == "train"].copy()
        dev_data = splits[splits.split == "dev"].copy()
        test_data = splits[splits.split == "test"].copy()
        
        self.train_dataset = Dataset(
            data_frame=self.train_data,
            video_path_prefix=f"{self.video_path}/0",
            frame_number=frame_number,
            frame_size=frame_size,
            transform=transform,
            augmentation=train_aug,
        )
        self.val_dataset = Dataset(
            data_frame=dev_data,
            video_path_prefix=f"{self.video_path}/0",
            frame_number=frame_number,
            frame_size=frame_size,
            transform=transform
        )
        self.test_dataset = Dataset(
            data_frame=test_data,
            video_path_prefix=f"{self.video_path}/0",
            frame_number=frame_number,
            frame_size=frame_size,
            transform=transform
        )

        self.num_sets = len(os.listdir(self.video_path))
        self.epoch = 0
        self.batch_size = batch_size

    def train_dataloader(self):
        set_idx = self.epoch % self.num_sets
        self.train_dataset._video_path_prefix = f"{self.video_path}/{set_idx}"
        self.epoch += 1
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.RandomSampler(self.train_dataset),
            num_workers=4,
            prefetch_factor=2,
            worker_init_fn=set_seed
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(self.val_dataset),
            num_workers=4,
            prefetch_factor=2,
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