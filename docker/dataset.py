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
        filename = data_row["filename"]
        index = data_row["row_index"]
        source = data_row["source"]
        index_to_read = index * 3 * self._num_eval_iters + 3 * self._epoch

        if source not in self._data_files:
            self._data_files[source] = mx.recordio.MXIndexedRecordIO(
                f'{source}.idx', 
                f'{source}.rec', 
                'r'
            )

        filename = data_row["filename"]
        label = int(data_row["label"])
        
        try:
            binary_faces = self._data_files[source].read_idx(index_to_read + 0)
            mel = pickle.loads(self._data_files[source].read_idx(index_to_read + 1))
            fn = self._data_files[source].read_idx(index_to_read + 2).decode()
            assert fn == filename

            with open(f"{index}", "wb") as f:
                f.write(binary_faces)
            faces = [face for face in VideoFileClip(f"{index}").iter_frames()]
            os.system(f"rm {index}")

            indices = list(range(len(faces)))
            np.random.shuffle(indices)
            indices = indices[:self._video_len]
            faces = [faces[i] for i in sorted(indices)]
            faces = np.array([
                cv2.resize(face, (self._video_size, self._video_size)) for face in faces
            ])
            if self._augmentation is not None:
                faces = np.array([self._augmentation(image = face)["image"] for face in faces])

            #plt.imsave(f"/data/temp/mel.png", mel)
            #for _, face in enumerate(faces):
            #    cv2.imwrite("/data/temp/face.png", face)            
            if mel is None or len(mel) == 0:
                mel = torch.zeros((self._audio_len, self._audio_size))
            else:
                mel = cv2.resize(mel, (self._audio_len, self._audio_size))
                mel = torch.tensor(mel).transpose(1,0)
                mel = (mel - torch.mean(mel))/torch.std(mel)
            mel = mel.reshape((self._video_len, self._audio_len//self._video_len, self._audio_size))
        except Exception as e:
            log(e)
            traceback.print_exc()
        
        sample_dict = {
            "video": torch.permute(torch.tensor(faces), (3, 0, 1, 2)),
            "audio": mel.half(),
            "label": label,
            "video_index": video_index,
            "filename": filename
        }

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)
        return sample_dict