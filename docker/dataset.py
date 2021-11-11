from __future__ import annotations
import traceback

from typing import Any, Callable, Optional
import time
import pickle
import numpy as np
import torch
from moviepy.editor import VideoFileClip
import torchaudio.transforms as T
import mxnet as mx
import os
from utils import log
import cv2
import torch.utils.data
import matplotlib.pylab as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_frame,
        epoch,
        video_len,
        video_size,
        audio_len,
        audio_size,
        resample_rate,
        num_eval_iters,
        freq_num,
        transform,
        augmentation = None,
    ) -> None:
        self._epoch = epoch
        self._data_frame = data_frame
        self._transform = transform
        self._video_len = video_len
        self._video_size = video_size
        self._audio_len = audio_len
        self._audio_size = audio_size
        self._augmentation = augmentation
        self._resample_rate = resample_rate
        self._num_eval_iters = num_eval_iters
        self._data_files = {}

        self._mel_spectrogram = T.MelSpectrogram(
            sample_rate=resample_rate,
            n_fft=2048,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=freq_num,
            mel_scale="htk",
        )
        
    def __len__(self):
        return len(self._data_frame)
    
    def __getitem__(self, video_index):
        data_row = self._data_frame.iloc[video_index]
        file_path = data_row["file_path"]
        label = int(data_row["label"])
        
        try:
            metadata = np.load(file_path)
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

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)
        return sample_dict