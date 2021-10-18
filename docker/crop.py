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
import torchaudio.transforms as T        
from retinaface.predict_single import Model
from utils import spectrogram, mel_spectrogram, mfcc_transform

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
queue = JoinableQueue()
res_queue = JoinableQueue()
fail_queue = JoinableQueue()


def apply_crop(frame, boxes, probs, crop_threshold=0.975, margin=0.5):
    if boxes is None:
        return frame, -1

    boxes = [box  for box, prob in zip(boxes, probs) if prob > crop_threshold]
    probs = [prob for prob in probs if prob > crop_threshold]

    if not boxes:
        return frame, -1

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


def crop_faces(face_detector, frames, skip_num=4, crop_batch_size=6):
    all_boxes, all_probs = [], []

    indices = range(0, len(frames), skip_num)
    select_frames = frames[indices, :, :, :]

    for i in range(int(np.ceil(len(select_frames)/crop_batch_size))):
        batch = select_frames[i*crop_batch_size: (i+1)*crop_batch_size]
        results = face_detector.predict_jsons_batch(batch)
        boxes = [[box["bbox"]  for box in frame_result] for frame_result in results]
        probs = [[box["score"] for box in frame_result] for frame_result in results]

        all_boxes += list(boxes)
        all_probs += list(probs)

    return_boxes, return_probs = [], []
    for i, _ in enumerate(frames):
        j = i // skip_num
        distance = 0
        boxes, probs = [], []
        while not boxes:
            l = max(0, j - distance)
            r = min(j + distance, len(all_probs)-1)
            boxes = all_boxes[l] or all_boxes[r]
            probs = all_probs[l] or all_probs[r]
            distance += 1

        return_boxes += [boxes]
        return_probs += [probs]
    return return_boxes, return_probs


def save(file_path, ouput_path, start, end, all_boxes, all_probs, resample_rate):
    audioclip = AudioFileClip(file_path).subclip(start, end)
    videoclip = VideoFileClip(file_path).subclip(start, end)

    video = []
    for frame in videoclip.iter_frames():
        video += [frame]
    video = np.array(video)

    audio = audioclip.to_soundarray()[:, 0]
    audio_rate = int(len(audio)/(end - start))
    audio = torch.tensor(librosa.resample(
        audio,
        audio_rate,
        resample_rate
    )).float()

    spec = librosa.power_to_db(spectrogram(audio))
    mel  = librosa.power_to_db(mel_spectrogram(audio))
    mfcc = librosa.power_to_db(mfcc_transform(audio))

def extract_video_audio():
    while True:
        try:
            results = res_queue.get()
        except Exception as e:
            log(e)
            continue

        if results is None:
            res_queue.task_done()
            return

        file_path, ouput_path, start, end, all_boxes, all_probs = results
        np.savez_compressed(
            ouput_path, 
            start=start,
            end=end,
            all_boxes=all_boxes,
            all_probs=all_probs,
        )
        save(file_path, ouput_path, start, end, all_boxes, all_probs)
        res_queue.task_done()
    

def worker(
    output_dir, 
    num_iters, 
    device, 
    max_clip_len,
    skip_num,
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

                    audioclip = audioclip.subclip(start, end)
                    videoclip = videoclip.subclip(start, end)
                    
                    video = []
                    for frame in videoclip.iter_frames():
                        video += [frame]
                    video = np.array(video)

                    all_boxes, all_probs = crop_faces(face_detector, video, skip_num)
                    ouput_path = f"{output_dir}/{file_name}_{iteration}"
                    res_queue.put((file_path, ouput_path, start, end, all_boxes, all_probs))
                    
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
            config['num_samples_per_video'],
            config['device'],
            config['max_clip_len'],
            config['face_detection_step'],
            args.no_cache,
        ))
        process.start()
        workers.append(process)

    for _ in range(2 * args.workers):
        extractor = Process(target=extract_video_audio)
        extractor.start()
        workers.append(extractor)

    queue.join()
    res_queue.join()

    for _ in range(args.workers):
        queue.put(None)

    for _ in range(2 * args.workers):
        res_queue.put(None)

    for w in workers:
        w.join()
    log(f'crop time: {time.time() - tik}')
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers",    type=int,   required=True, help="Number of workers")
    parser.add_argument("--input",      type=str,   required=True, help="Input directory of raw videos")
    parser.add_argument("--output",     type=str,   required=True, help="Output directory of faces")
    parser.add_argument("--no-cache",   action='store_true')

    args = parser.parse_args()
    main(args)