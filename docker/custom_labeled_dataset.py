from __future__ import annotations

import logging
from typing import Any, Callable, Optional
import cv2
import numpy as np
import torch
import pickle
import os

import torch.utils.data

from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths


logger = logging.getLogger(__name__)


class CustomVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        video_path_prefix: str = "",
        transform: Optional[Callable[[dict], Any]] = None,
        frame_number: int = 32,
        frame_size: int = 224,
        augmentation = None,
    ) -> None:
        self._transform = transform
        self._frame_number = frame_number
        self._frame_size = frame_size
        self._labeled_videos = LabeledVideoPaths.from_path(data_path)
        self._labeled_videos.path_prefix = video_path_prefix
        self._augmentation = augmentation
        
    def __len__(self):
        return len(self._labeled_videos)
    
    def __getitem__(self, video_index):
        video_index = video_index % len(self._labeled_videos)
        video_path, info_dict = self._labeled_videos[video_index]

        frames = np.load(f"{video_path}.npy")
        missing_frames = self._frame_number - len(frames)
        frames = torch.tensor(np.stack(frames))
        frames = torch.nn.functional.pad(
            frames, 
            (0, 0, 0, 0, 0, 0, 0, missing_frames), 
        )
        frames = frames.permute(3, 0, 1, 2)

        sample_dict = {
            "video":        frames,
            "video_index":  video_index,
            **info_dict,
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict