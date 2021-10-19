import cv2
import os
import matplotlib.pylab as plt
from multiprocessing import Process, JoinableQueue
import traceback
import json
import time
import numpy as np
from argparse import ArgumentParser
from utils import log, create_folder
import librosa
import torch
from moviepy.editor import AudioFileClip, VideoFileClip
import torch        
import warnings
import torchaudio.transforms as T        
from retinaface.predict_single import Model

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
queue = JoinableQueue()
fail_queue = JoinableQueue()


def apply_crop(frame, boxes, probs, margin=0.5):
    face_idx = np.random.randint(len(boxes))
    x, y, z, t = boxes[face_idx]
    x, y, z, t = int(x), int(y), int(z), int(t)
    width  = x - z
    height = t - y
    center_y = int((y + t) / 2)
    center_x = int((x + z) / 2)
    size = max(width, height)
    size = int(size * (1 + margin) / 2)

    y = max(0, center_y - size)
    t = center_y + size
    x = max(0, center_x - size)
    z = center_x + size

    noise = int(0.05 * size)
    y += np.random.randint(-noise, noise+1)
    t += np.random.randint(-noise, noise+1)
    x += np.random.randint(-noise, noise+1)
    z += np.random.randint(-noise, noise+1)
    y = max(0, y)
    x = max(0, x)
    return frame[y:t, x:z, :], probs[face_idx]


def crop_faces(face_detector, frames, skip_num=4, crop_batch_size=6, crop_threshold=0.975):
    all_boxes, all_probs = [], []

    indices = range(0, len(frames), skip_num)
    select_frames = frames[indices, :, :, :]

    for i in range(int(np.ceil(len(select_frames)/crop_batch_size))):
        batch = select_frames[i*crop_batch_size: (i+1)*crop_batch_size]
        results = face_detector.predict_jsons_batch(batch)
        boxes = [[box["bbox"]  for box in frame_result if box["score"] > crop_threshold] for frame_result in results]
        probs = [[box["score"] for box in frame_result if box["score"] > crop_threshold] for frame_result in results]

        all_boxes += boxes
        all_probs += probs

    return_boxes, return_probs = [], []
    for i in range(len(frames)):
        j = i // skip_num
        distance = 0
        boxes, probs = [], []
        while not boxes:
            l = max(0, j - distance)
            r = min(j + distance, len(all_probs)-1)
            boxes = all_boxes[l] or all_boxes[r]
            probs = all_probs[l] or all_probs[r]
            
            if l == 0 and r == len(all_probs)-1 and not boxes:
                log("No faces found. Use raw frames!")
                w, h = frames[0].shape[1:3]
                return_boxes = [[[0, 0, h, w]] for _ in range(len(frames))]
                return_probs = [[1] for _ in range(len(frames))]
                return return_boxes, return_probs

            distance += 1

        return_boxes += [boxes]
        return_probs += [probs]
    return return_boxes, return_probs


def extract_audio_video(
    file_path, 
    start, 
    end, 
    all_boxes, 
    all_probs, 
    resample_rate,
    spectrogram,
    mel_spectrogram,
    mfcc_transform,    
):
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        audioclip = AudioFileClip(file_path).subclip(start, end)
        videoclip = VideoFileClip(file_path).subclip(start, end)

        select_faces, select_probs = [], []
        for frame, boxes, probs in zip(videoclip.iter_frames(), all_boxes, all_probs):
            try:
                face, prob = apply_crop(frame, boxes, probs)
            except:
                print(file_path, boxes, probs)
            select_faces += [face]
            select_probs += [prob]

        try:
            audio = audioclip.to_soundarray()
            audio = audio[:, 0]
            audio_rate = int(len(audio)/(end - start))
            audio = torch.tensor(librosa.resample(
                audio,
                audio_rate,
                resample_rate
            )).float()

            spec = librosa.power_to_db(spectrogram(audio))
            mel  = librosa.power_to_db(mel_spectrogram(audio))
            mfcc = librosa.power_to_db(mfcc_transform(audio))
        except Exception as e:
            log("No audio found!")
            spec, mel, mfcc = [None for _ in range(3)]

        faces = [face for (face, prob) in zip(select_faces, select_probs) if prob > 0]
        probs = [prob for prob in select_probs if prob > 0]
        return faces, probs, spec, mel, mfcc


def worker(
    output_dir,
    save_image,
    num_iters, 
    device, 
    max_clip_len,
    skip_num,
    freq_num,
    resample_rate,
    no_cache,
):    
    try:
        face_detector = Model(
            max_size=2048, 
            device=device
        )        
        state_dict = torch.load("retinaface_resnet50_2020-07-20.zip")
        face_detector.load_state_dict(state_dict)
        face_detector.eval()

        spectrogram = T.Spectrogram(
            n_fft=2*freq_num-1,
            hop_length=1024,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        mel_spectrogram = T.MelSpectrogram(
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
        mfcc_transform = T.MFCC(
            sample_rate=resample_rate,
            n_mfcc=freq_num,
            melkwargs={
            'n_fft': 2048,
            'n_mels': freq_num,
            'mel_scale': 'htk',
            }
        )
    except Exception as e:
        log(e)
        return

    while True:
        file_path = queue.get()
        
        if file_path is None:
            queue.task_done()
            return

        if fail_queue.qsize() > 8:
            queue.task_done()
            continue

        file_name = file_path.split("/")[-1]

        try:
            if no_cache or not os.path.exists(f"{output_dir}/{file_name}_{num_iters-1}.npz"):
                audioclip = AudioFileClip(file_path)
                videoclip = VideoFileClip(file_path)

                for iteration in range(num_iters):
                    duration = audioclip.duration
                    start = max(0, duration-max_clip_len) * np.random.rand()
                    end = min(start + max_clip_len, duration)

                    clip = videoclip.subclip(start, end)
                    video = []
                    for frame in clip.iter_frames():
                        video += [frame]
                    video = np.array(video)

                    all_boxes, all_probs = crop_faces(face_detector, video, skip_num)
                    output_path = f"{output_dir}/{file_name}_{iteration}"
                    np.savez_compressed(
                        output_path,
                        start=start,
                        end=end,
                        all_boxes=all_boxes,
                        all_probs=all_probs,
                    )
                    if save_image:
                        faces, probs, spec, mel, mfcc = extract_audio_video(
                            file_path, 
                            start, 
                            end, 
                            all_boxes, 
                            all_probs, 
                            resample_rate,
                            spectrogram,
                            mel_spectrogram,
                            mfcc_transform,  
                        )
                        create_folder(output_path)
                        for i, (face, prob) in enumerate(zip(faces, probs)):
                            cv2.imwrite(f"{output_path}/{i}_{prob}.png", face)
                        if spec is not None:
                            plt.imsave(f"{output_path}/spec.png", spec)
                            plt.imsave(f"{output_path}/mel.png", mel)
                            plt.imsave(f"{output_path}/mfcc.png", mfcc)
                    
        except Exception as e:
            queue.put(file_path)
            fail_queue.put(file_path)
            log("=" * 20)
            log(e)
            log(f"number of failed videos: {fail_queue.qsize()}")
            traceback.print_exc()
        queue.task_done()
        

def main(args):
    tik = time.time()
    config = json.load(open("config.json"))
    create_folder(args.output)

    for f in os.listdir(args.input):
        if f.endswith(".mp4"):
            queue.put(f"{args.input}/{f}")

    workers = []
    for _ in range(args.workers):
        process = Process(target=worker, args=(
            args.output,
            args.save_image,
            config['num_samples_per_video'],
            config['device'],
            config['max_clip_len'],
            config['face_detection_step'],
            config['freq_num'],
            config['resample_rate'],
            args.no_cache,
        ))
        process.start()
        workers.append(process)

    queue.join()

    for _ in range(args.workers):
        queue.put(None)

    for w in workers:
        w.join()
    log(f'crop time: {time.time() - tik}')
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers",    type=int,   required=True, help="Number of workers")
    parser.add_argument("--input",      type=str,   required=True, help="Input directory of raw videos")
    parser.add_argument("--output",     type=str,   required=True, help="Output directory of faces")
    parser.add_argument("--save-image", action='store_true')
    parser.add_argument("--no-cache",   action='store_true')

    args = parser.parse_args()
    main(args)