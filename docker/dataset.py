from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np
import torch
from utils import log
import cv2
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_frame,
        video_path_prefix: str = "",
        transform: Optional[Callable[[dict], Any]] = None,
        frame_number: int = 32,
        augmentation = None,
    ) -> None:
        self._epoch = 0
        self._data_frame = data_frame
        self._transform = transform
        self._frame_number = frame_number
        self._video_path_prefix = video_path_prefix
        self._augmentation = augmentation        
        
    def __len__(self):
        return len(self._data_frame)
    
    def __getitem__(self, video_index):
        data_row = self._data_frame.iloc[video_index]
        filename = data_row["filename"] 
        label = int(data_row["label"])

        raw_frames = np.load(f"{self._video_path_prefix}/{filename}.npy")
        frames = []
        for _, raw_frame in enumerate(raw_frames):
            frame = self._augmentation(image = raw_frame)["image"]
            frames += [frame]
            #cv2.imwrite(f"/data/temp/{filename}_{_}.png", frame)
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
            "video_index": video_index,
            "filename": filename
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)
        return sample_dict