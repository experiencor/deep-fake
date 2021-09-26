import cv2
import os
import torch
from retinaface.pre_trained_models import get_model
from multiprocessing import Process, JoinableQueue
import traceback
import json
import time
import numpy as np
from argparse import ArgumentParser
from utils import log, create_folder
import hashlib

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
queue = JoinableQueue()
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
    return frame[y:t, x:z, :], probs[face_idx]


def crop_faces(face_detector, frames, frame_size, crop_batch_size=8):
    all_boxes, all_probs = [], []

    for i in range(int(np.ceil(len(frames)/crop_batch_size))):
        batch = frames[i*crop_batch_size: (i+1)*crop_batch_size]
        results = face_detector.predict_jsons_batch(batch)
        boxes = [[box["bbox"]  for box in frame_result] for frame_result in results]
        probs = [[box["score"] for box in frame_result] for frame_result in results]

        all_boxes += list(boxes)
        all_probs += list(probs)

    return_faces, return_probs = [], []
    for frame, boxes, probs in zip(frames, all_boxes, all_probs):
        face, prob = apply_crop(frame, boxes, probs)
        face = cv2.resize(face, (frame_size, frame_size))
        return_faces += [face]
        return_probs += [prob]
    
    return return_faces, return_probs
    

def worker(output_dir, save_image, no_cache, num_iters, device, frame_number, frame_size):
    try:
        face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
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
        seed = hashlib.sha224(file_name.encode('utf-8')).hexdigest()
        seed = int(seed[16:20], 16)
        np.random.seed(seed)        

        try:
            if no_cache or not (os.path.exists(f"{output_dir}/{num_iters-1}/{file_name}.npy") or os.path.exists(f"{output_dir}/{num_iters-1}/{file_name}.npz")):
                video = cv2.VideoCapture(file_path)
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                total_count = num_iters * frame_number

                all_the_indices = np.arange(frame_count)
                np.random.shuffle(all_the_indices)
                selected_indices = []
                while len(all_the_indices) > 0:
                    buffer_size     = int(1.2 * total_count)
                    chosen_indices  = all_the_indices[:buffer_size]
                    all_the_indices = all_the_indices[buffer_size:]
                    chosen_indices  = sorted(chosen_indices)

                    frames = []
                    i, j, _ = 0, 0, video.grab()
                    while len(frames) < len(chosen_indices):
                        if i == chosen_indices[j]:
                            _, frame = video.retrieve()
                            frames += [frame]
                            j += 1
                        _ = video.grab()
                        i += 1

                    faces, probs = crop_faces(face_detector, frames, frame_size)
                    for index, face, prob in zip(chosen_indices, faces, probs):
                        if prob > 0:
                            selected_indices += [[index, face, prob]]

                    if len(selected_indices) >= total_count:
                        break
                    video = cv2.VideoCapture(file_path)
                else:
                    log(f"found {len(selected_indices)} but required {total_count} for {file_name}")

                selected_indices = sorted(selected_indices)
                detect_faces = [face for _, face, _ in selected_indices[:total_count]]
                detect_probs = [prob for _, _, prob in selected_indices[:total_count]]

                random_indices  = np.arange(len(detect_faces))
                l_idx, r_idx    = 0, frame_number
                increment       = (len(detect_faces) + 1 - frame_number) // num_iters
                np.random.shuffle(random_indices)
                for iteration in range(num_iters):
                    iter_path = f"{output_dir}/{iteration}"
                    create_folder(iter_path)
                    
                    select_indices = random_indices[l_idx:r_idx]
                    select_indices = sorted(select_indices)

                    iter_faces = np.array([detect_faces[i] for i in select_indices])
                    iter_probs = np.array([detect_probs[i] for i in select_indices])
                    np.savez_compressed(f"{iter_path}/{file_name}", faces=iter_faces)

                    if save_image:
                        image_path = f"{iter_path}/{file_name}"
                        create_folder(image_path)
                        for i, (face, prob) in enumerate(zip(iter_faces, iter_probs)):
                            cv2.imwrite(f"{image_path}/{i}_{prob}.png", face)

                    l_idx += increment
                    r_idx  = l_idx + frame_number
        except Exception as e:
            queue.put(file_path)
            fail_queue.put(file_path)
            log("=" * 20)
            log(e)
            log(f"number of failed videos: {fail_queue.qsize()}")
            traceback.print_exc()

        np.random.seed()
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
            args.no_cache,
            config['num_eval_iters'],
            config['device'],
            config['frame_number'],
            config['frame_size']
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