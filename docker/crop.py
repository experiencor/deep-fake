import cv2
import os
import matplotlib.pylab as plt
from multiprocessing import Process, JoinableQueue
import traceback
import json
import mxnet as mx
from moviepy.video.io import ImageSequenceClip
import time
from retinaface.pre_trained_models import get_model
import numpy as np
from argparse import ArgumentParser
from utils import log, create_folder
import time
import librosa
import pickle
import torch
from moviepy.editor import AudioFileClip, VideoFileClip
import torch        
import torchaudio.transforms as T        
from retinaface.predict_single import Model

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
queue = JoinableQueue()
extraction_queue = JoinableQueue()
output_queue = JoinableQueue()
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


def crop_faces(file_path, face_detector, frames, skip_num=4, crop_batch_size=8, crop_threshold=0.975):
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
                log("No faces found. Use raw frames!", file_path)
                w, h = frames[0].shape[1:3]
                return_boxes = [[[0, 0, h, w]] for _ in range(len(frames))]
                return_probs = [[1] for _ in range(len(frames))]
                return return_boxes, return_probs

            distance += 1

        return_boxes += [boxes]
        return_probs += [probs]
    return return_boxes, return_probs


def extract(extractor_index, freq_num, resample_rate, num_iters):
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

    while True:
        print(">" * 60, "extraction_queue", extraction_queue.qsize())
        elem = extraction_queue.get()

        if elem == None:
            extraction_queue.task_done()
            return

        file_path, index, iteration, save_image, start, end, select_faces, select_probs = elem
        file_name = file_path.split("/")[-1]
        
        # print(f"from {extractor_index}, \textracting audio")
        try:
            audio = AudioFileClip(file_path).subclip(start, end).to_soundarray()
            audio = audio[:, 0]
            audio_rate = int(len(audio)/(end - start))
            audio = torch.tensor(librosa.resample(
                audio,
                audio_rate,
                resample_rate
            )).float()

            spec = librosa.power_to_db(spectrogram(audio))     if spectrogram       else None
            mel  = librosa.power_to_db(mel_spectrogram(audio)) if mel_spectrogram   else None
            mfcc = librosa.power_to_db(mfcc_transform(audio))  if mfcc_transform    else None
        except Exception as e:
            log("No audio found!", e)
            audio = np.array([])
            spec, mel, mfcc = [np.array([]) for _ in range(3)]

        # print(f"from {extractor_index}, \tproducing the mp4")
        index_to_write = index * (3 * num_iters) + 3 * iteration
        clip = ImageSequenceClip.ImageSequenceClip(
            select_faces, 
            fps=25
        )

        # print(f"from {extractor_index}, \tadd items to output queue")
        clip.write_videofile(f"{extractor_index}.mp4", verbose=False)
        with open(f"{extractor_index}.mp4", "rb") as f:
            output_queue.put((
                index_to_write + 0,
                f.read()
            ))
        os.system(f"rm {extractor_index}.mp4")

        output_queue.put((
            index_to_write + 1,
            pickle.dumps(mel)
        ))

        output_queue.put((
            index_to_write + 2,
            file_name.encode()
        ))

        # print(f"from {extractor_index}, \tsaving the data")
        if save_image:
            create_folder("temp")
            visible_folder = f"temp/{index}_{iteration}/"
            create_folder(visible_folder)
            for i, (face, prob) in enumerate(zip(select_faces, select_probs)):
                cv2.imwrite(f"{visible_folder}/{i}_{prob}.png", face)

            if spec is not None:
                plt.imsave(f"{visible_folder}/spec.png", spec)

            if mel is not None:
                plt.imsave(f"{visible_folder}/mel.png", mel)

            if mfcc is not None:
                plt.imsave(f"{visible_folder}/mfcc.png", mfcc)

        extraction_queue.task_done()
        # print(f"from {extractor_index}, \tcompleted")


def output(output_file):
    records = mx.recordio.MXIndexedRecordIO(
        f'{output_file}.idx', f'{output_file}.rec', 'w'
    )

    while True:
        print(">" * 60, "output_queue", output_queue.qsize())
        elem = output_queue.get()

        if elem == None:
            records.close()
            output_queue.task_done()
            return

        index, data = elem
        records.write_idx(
            index, 
            data
        )
        output_queue.task_done()


def work(
    save_image,
    num_iters, 
    device, 
    max_clip_len,
    skip_num,
    video_size
):    
    try:
        face_detector = get_model(
            model_name="resnet50_2020-07-20",
            max_size=2048, 
            device=device
        )        
        face_detector.eval()
    except Exception as e:
        log(e)
        return

    while True:
        print(">" * 60, "queue", queue.qsize())
        elem = queue.get()
        
        if elem is None:
            queue.task_done()
            return

        if fail_queue.qsize() > 8:
            queue.task_done()
            continue

        index, file_path = elem

        try:
            videoclip = VideoFileClip(file_path)
            duration = videoclip.duration

            for iteration in range(num_iters):
                start = max(0, duration-max_clip_len) * np.random.rand()
                end = min(start + max_clip_len, duration)

                frames = []
                for frame in videoclip.subclip(start, end).iter_frames():
                    frames += [frame]

                indices = list(range(len(frames)))
                np.random.shuffle(indices)
                indices = indices[:96]
                frames = [frames[i] for i in sorted(indices)]
                frames = np.array(frames)

                all_boxes, all_probs = crop_faces(file_path, face_detector, frames, skip_num)
                select_faces, select_probs = [], []
                for frame, boxes, probs in zip(frames, all_boxes, all_probs):
                    face, prob = apply_crop(frame, boxes, probs)
                    select_faces += [cv2.resize(face, (video_size, video_size))]
                    select_probs += [prob]

                extraction_queue.put((
                    file_path, 
                    index, 
                    iteration, 
                    save_image, 
                    start, 
                    end, 
                    select_faces, 
                    select_probs
                ))
        except Exception as e:
            queue.put(file_path)
            fail_queue.put((index, file_path))
            log("=" * 20)
            log(e)
            log(f"number of failed videos: {fail_queue.qsize()}")
            traceback.print_exc()
        queue.task_done()
        

def main(args):
    tik = time.time()
    config = json.load(open("config.json"))
    create_folder(args.output)

    index = 0
    files = [f for f in sorted(os.listdir(args.input)) if f.endswith(".mp4")]
    for f in files:
        queue.put((index, f"{args.input}/{f}"))
        index += 1

    # start the workers
    workers = []
    for _ in range(args.num_workers):
        worker = Process(target=work, args=(
            args.save_image,
            config['num_samples_per_video'],
            config['device'],
            config['max_clip_len'],
            config['face_detection_step'],
            config['video_size'],
        ))
        worker.start()
        workers.append(worker)

    extractors = []
    for i in range(args.num_workers):
        extractor = Process(target=extract, args=(
            i,
            config['freq_num'],
            config['resample_rate'],
            config['num_samples_per_video'],            
        ))
        extractor.start()
        extractors += [extractor]

    outputer = Process(target=output, args=(
        args.output,
    ))
    outputer.start()

    # wait for the queue to be empty
    queue.join()
    extraction_queue.join()
    output_queue.join()

    # put None stop the workers
    for _ in range(args.num_workers):
        queue.put(None)

    for _ in range(args.num_workers):
        extraction_queue.put(None)

    output_queue.put(None)

    # wait for the workers to finish
    for worker in workers:
        worker.join()

    for extractor in extractors:
        extractor.join()

    outputer.join()
    log(f'average crop time: {(time.time() - tik)/index}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-workers",    type=int,   required=True, help="Number of workers")
    parser.add_argument("--input",          type=str,   required=True, help="Input directory of raw videos")
    parser.add_argument("--output",         type=str,   required=True, help="Output file of face and audio")
    parser.add_argument("--save-image",     action='store_true')

    args = parser.parse_args()
    main(args)