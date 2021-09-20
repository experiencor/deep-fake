from facenet_pytorch import MTCNN
import cv2
import os
import torch
from facenet_pytorch import MTCNN
import logging
from multiprocessing import Process, JoinableQueue
import pickle
import traceback
import numpy as np
from argparse import ArgumentParser
from utils import log, create_folder
import hashlib
from eval import FRAME_NUMBER, FRAME_SIZE, CROP_BATCH_SIZE, CROP_THRESHOLD, DEVICE

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
queue = JoinableQueue()
fail_queue = JoinableQueue()


def apply_crop(frame, boxes, probs, margin=0.5):
    if boxes is None:
        return frame, 0

    boxes = [box for box, prob in zip(boxes, probs) if prob > CROP_THRESHOLD]
    probs = [prob for prob in probs if prob > CROP_THRESHOLD]

    if not boxes:
        return frame, 0

    face_idx = np.random.randint(len(boxes))
    x, y, z, t = boxes[face_idx]
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


def crop_faces(mtcnn, frames):
    all_boxes, all_probs = [], []

    for i in range(int(np.ceil(len(frames)/CROP_BATCH_SIZE))):
        batch = frames[i*CROP_BATCH_SIZE: (i+1)*CROP_BATCH_SIZE]
        boxes, probs = mtcnn.detect(batch)
        all_boxes += list(boxes)
        all_probs += list(probs)

    return_faces, return_probs = [], []
    for frame, boxes, probs in zip(frames, all_boxes, all_probs):
        face, prob = apply_crop(frame, boxes, probs)
        face = cv2.resize(face, (FRAME_SIZE, FRAME_SIZE))
        return_faces += [face]
        return_probs += [prob]
    
    return return_faces, return_probs
    

def worker(output_dir, num_iters, save_image, no_cache):
    mtcnn = MTCNN(
        margin = 14,
        factor = 0.6,
        keep_all = True,
        device = DEVICE if torch.cuda.is_available() else "cpu"
    )
    
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
            if not os.path.exists(f"{output_dir}/0/{file_name}.npy") or no_cache:
                video = cv2.VideoCapture(file_path)
                
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                total_count = num_iters * FRAME_NUMBER
                if total_count <= frame_count:
                    chosen_indices = np.random.choice(range(frame_count), total_count, replace=False)
                else:
                    chosen_indices = np.range(frame_count)
                chosen_indices = sorted(chosen_indices)

                frames = []
                i, j, status = 0, 0, video.grab()
                while len(frames) < total_count and status:
                    if i == chosen_indices[j]:
                        _, frame = video.retrieve()
                        frames += [frame]
                        j += 1
                    status = video.grab()
                    i += 1

                faces, probs = crop_faces(mtcnn, frames)

                iter_face_count = len(faces) // num_iters
                random_indices  = np.arange(len(faces))
                l_idx, r_idx    = 0, FRAME_NUMBER
                increment       = (len(faces) + 1 - FRAME_NUMBER) // num_iters
                np.random.shuffle(random_indices)
                for iteration in range(num_iters):
                    iter_path = f"{output_dir}/{iteration}"
                    create_folder(iter_path)
                    
                    select_indices = random_indices[l_idx:r_idx]
                    select_indices = sorted(select_indices)

                    iter_faces = np.array([faces[i] for i in select_indices])
                    iter_probs = np.array([probs[i] for i in select_indices])
                    np.save(f"{iter_path}/{file_name}.npy", iter_faces)

                    if save_image:
                        for i, (face, prob) in enumerate(zip(iter_faces, iter_probs)):
                            image_path = f"{iter_path}/{file_name}"
                            create_folder(image_path)
                            cv2.imwrite(f"{image_path}/{i}_{prob}.png", face)

                    l_idx += increment
                    r_idx  = l_idx + FRAME_NUMBER
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
    create_folder(args.output)

    for f in os.listdir(args.input):
        if f.endswith(".mp4"):
            queue.put(f"{args.input}/{f}")

    workers = []
    for i in range(args.workers):
        process = Process(target=worker, args=(args.output, args.num_iters, args.image, args.no_cache))
        process.start()
        workers.append(process)
    queue.join()

    for i in range(args.workers):
        queue.put(None)

    for w in workers:
        w.join()
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers",    type=int,  required=True, help="Number of workers")
    parser.add_argument("--num-iters",  type=int,  required=True, help="Number of iterations")
    parser.add_argument("--input",      type=str,  required=True, help="Input directory of raw videos")
    parser.add_argument("--output",     type=str,  required=True, help="Output directory of faces")
    parser.add_argument("--image",      action='store_true')
    parser.add_argument("--no-cache",   action='store_true')

    args = parser.parse_args()
    main(args)