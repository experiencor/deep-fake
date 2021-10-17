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
        frame_num,
        freq_num,
        audio_len,
        epoch: int = 0,
        transform: Optional[Callable[[dict], Any]] = None,
        frame_number: int = 32,
        augmentation = None,
    ) -> None:
        self._epoch = epoch
        self._frame_size = frame_size
        self._frame_num = frame_num
        self._data_frame = data_frame
        self._transform = transform
        self._frame_number = frame_number
        self._augmentation = augmentation
        self._freq_num = freq_num
        self._audio_len = audio_len     
        
    def __len__(self):
        return len(self._data_frame)
    
    def __getitem__(self, video_index):
        data_row = self._data_frame.iloc[video_index]
        prefix = data_row["prefix"]
        filename = data_row["filename"]
        label = int(data_row["label"])
        
        try:
            metadata = np.load(f"{prefix}/{filename}_{self._epoch}.npz", allow_pickle=True)
            faces = metadata["faces"]
            faces = np.array([cv2.resize(face, (self._frame_size, self._frame_size)) for face in faces])
            indices = list(range(len(faces)))
            np.random.shuffle(indices)
            faces = faces[sorted(indices),:,:,:]

            mel = metadata["mel"]
            mel = cv2.resize(mel, (self._freq_num, self._audio_len))
            if self._augmentation is None:
                faces = list(faces)
            else:
                faces = [self._augmentation(image = face)["image"] for face in faces]
                #for _, frame in enumerate(frames):
                #    cv2.imwrite(f"/data/temp/{filename}_{_}.png", frame)
        except Exception as e:
            log(e)

        print(faces.shape, mel.shape)

        sample_dict = {
            "video": faces,
            "audio": mel,
            "label": label,
            "video_index": video_index,
            "filename": filename
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)
        return sample_dict