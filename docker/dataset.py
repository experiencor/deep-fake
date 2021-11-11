from __future__ import annotations
import traceback

import numpy as np
import torch
import torchaudio.transforms as T
from utils import log
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_frame,
        augmentation = None,
    ) -> None:
        self._data_frame = data_frame
        self._augmentation = augmentation
        
    def __len__(self):
        return len(self._data_frame)
    
    def __getitem__(self, video_index):
        data_row = self._data_frame.iloc[video_index]
        file_path = data_row["file_path"]
        file_name = data_row["file_name"].split(".")[0]
        label = int(data_row["label"])
        
        try:
            metadata = np.load(f"{file_path}/{file_name}.npz")
            frames = metadata["faces"] + metadata["mel_3"]

            if self._augmentation is not None:
                frames = np.array([self._augmentation(image = frame)["image"] for frame in frames])
        except Exception as e:
            log(e)
            traceback.print_exc()
        
        sample_dict = {
            "video": torch.permute(torch.tensor(frames), (3, 0, 1, 2)),
            "label": label,
            "video_index": video_index,
            "file_path": file_path
        }

        return sample_dict
