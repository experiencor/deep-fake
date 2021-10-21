from __future__ import annotations
import traceback

from typing import Any, Callable, Optional
import time
import numpy as np
import torch
import torchaudio.transforms as T
from utils import log
import cv2
import torch.utils.data
import matplotlib.pylab as plt
from crop import extract_audio_video


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
        freq_num,
        transform,
        audio_transform,
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
        self._audio_transform = audio_transform

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
        meta_prefix = data_row["meta_prefix"]
        video_prefix = data_row["video_prefix"]
        filename = data_row["filename"]
        label = int(data_row["label"])
        
        try:
            tik = time.time()
            metadata = np.load(f"{meta_prefix}/{filename}_{self._epoch}.npz", allow_pickle=True)
            print("\nload metadata", (time.time() - tik))

            tik = time.time()
            faces, _, _, mel, _ = extract_audio_video(
                f"{video_prefix}/{filename}", 
                metadata["start"], 
                metadata["end"], 
                metadata["all_boxes"], 
                metadata["all_probs"], 
                self._resample_rate,
                None,
                self._mel_spectrogram,
                None,  
            )
            print("\nextract a/v", (time.time() - tik))

            tik = time.time()
            indices = list(range(len(faces)))
            np.random.shuffle(indices)
            indices = indices[:self._video_len]
            faces = [faces[i] for i in sorted(indices)]
            faces = np.array([
                cv2.resize(face, (self._video_size, self._video_size)) for face in faces
            ])

            mel = cv2.resize(mel, (self._audio_len, self._audio_size))
            if self._augmentation is not None:
                faces = np.array([self._augmentation(image = face)["image"] for face in faces])
            print("\naugmentation", (time.time() - tik))

            #for _, face in enumerate(faces):
            #    cv2.imwrite(f"/data/temp/{filename}_{_}.png", face)
            #plt.imsave(f"/data/temp/mel.png", mel)
        except Exception as e:
            log(e)
            traceback.print_exc()

        mel = torch.tensor(mel).transpose(1,0)
        mel = (mel - torch.mean(mel))/torch.std(mel)

        sample_dict = {
            "video": torch.permute(torch.tensor(faces), (3, 0, 1, 2)),
            "audio": mel,
            "label": label,
            "video_index": video_index,
            "filename": filename
        }

        if self._transform is not None:
            sample_dict = self._transform(sample_dict)
        return sample_dict