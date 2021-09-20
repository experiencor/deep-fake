from __future__ import annotations

import logging
from typing import Any, Callable, Optional
import cv2
import numpy as np
import torch
import pickle
import os
from .utils import log

import torch.utils.data

from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths


logger = logging.getLogger(__name__)


class CustomVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_frame,
        video_path_prefix: str = "",
        transform: Optional[Callable[[dict], Any]] = None,
        frame_number: int = 32,
        frame_size: int = 224,
        augmentation = None,
    ) -> None:
        self._access_count = 0
        self._data_frame = data_frame
        self._transform = transform
        self._frame_number = frame_number
        self._frame_size = frame_size
        self._video_path_prefix = video_path_prefix
        self._augmentation = augmentation
        
    def __len__(self):
        log("=" * 25)
        return len(self._data_frame)
    
    def __getitem__(self, video_index):
        data_row = self._data_frame.iloc[video_index]
        filename = data_row["filename"] 
        label = int(data_row["label"])

        frames = np.load(f"{self._video_path_prefix}/{filename}.npy")
        missing_frames = self._frame_number - len(frames)
        frames = torch.tensor(np.stack(frames))
        frames = torch.nn.functional.pad(
            frames, 
            (0, 0, 0, 0, 0, 0, 0, missing_frames), 
        )
        frames = frames.permute(3, 0, 1, 2)

        sample_dict = {
            "video": frames,
            "label": label,
            "video_index": video_index
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)
        return sample_dict