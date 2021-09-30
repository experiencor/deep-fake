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
        frame_size,
        epoch: int = 0,
        transform: Optional[Callable[[dict], Any]] = None,
        frame_number: int = 32,
        augmentation = None,
    ) -> None:
        self._epoch = epoch
        self._frame_size = frame_size
        self._data_frame = data_frame
        self._transform = transform
        self._frame_number = frame_number
        self._augmentation = augmentation        
        
    def __len__(self):
        return len(self._data_frame)
    
    def __getitem__(self, video_index):
        data_row = self._data_frame.iloc[video_index]
        prefix = data_row["prefix"]
        filename = data_row["filename"] 
        label = int(data_row["label"])
        
        try:
            raw_frames = np.load(f"{prefix}/{self._epoch}/{filename}.npz")["faces"]
            print(f"{prefix}/{self._epoch}/{filename}.npz")
            if self._augmentation is None:
                frames = list(raw_frames)
            else:
                frames = [self._augmentation(image = raw_frame)["image"] for raw_frame in raw_frames]
                for _, frame in enumerate(frames):
                    cv2.imwrite(f"/data/temp/{filename}_{_}.png", frame)
        except Exception as e:
            frames = []
            log(e)

        pads   = [torch.zeros((self._frame_size, self._frame_size, 3)) for _ in range(self._frame_number - len(frames))]
        frames = torch.tensor(np.stack(frames + pads))
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