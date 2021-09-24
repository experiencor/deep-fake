import cv2
import os
import torch
from retinaface.pre_trained_models import get_model
import multiprocessing
from multiprocessing import Process, JoinableQueue, Queue
import traceback
import json
from torch.nn import functional as F
import time
import numpy as np
from argparse import ArgumentParser
from retinaface.box_utils import decode, decode_landm
from utils import log, create_folder
from torchvision.ops import nms
import hashlib
from retinaface.prior_box import priorbox

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
queue = JoinableQueue()
fail_queue = JoinableQueue()
detection_queue = JoinableQueue()
nms_queue = JoinableQueue()
m = multiprocessing.Manager()


ROUNDING_DIGITS = 2


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


def crop_faces(frames, frame_size, crop_batch_size=4):
    all_boxes, all_probs = [], []

    for i in range(int(np.ceil(len(frames)/crop_batch_size))):
        batch = frames[i*crop_batch_size: (i+1)*crop_batch_size]
        results_queue = m.Queue()
        detection_queue.put((batch, results_queue))

        results = []
        while len(results) < len(batch):
            results += [results_queue.get()]
        results.sort()

        boxes = [[box["bbox"]  for box in frame_result] for _, frame_result in results]
        probs = [[box["score"] for box in frame_result] for _, frame_result in results]

        all_boxes += list(boxes)
        all_probs += list(probs)

    return_faces, return_probs = [], []
    for frame, boxes, probs in zip(frames, all_boxes, all_probs):
        face, prob = apply_crop(frame, boxes, probs)
        face = cv2.resize(face, (frame_size, frame_size))
        return_faces += [face]
        return_probs += [prob]
    
    return return_faces, return_probs


def detect_faces(device):
    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    while True:
        print(f"detection_queue size: {detection_queue.qsize()}")
        frames, results_queue = detection_queue.get()
        print(f"frames: {len(frames)}")
        (bboxs, confs, lands), transformed_shape = face_detector.predict(frames)
        bboxs = bboxs.cpu().detach().to(torch.float32)
        confs = confs.cpu().detach().to(torch.float32)
        lands = lands.cpu().detach().to(torch.float32)

        for frame_id, (frame, loc, conf, land) in enumerate(zip(frames, bboxs, confs, lands)):
            nms_queue.put((frame_id, frame, loc, conf, land, transformed_shape, face_detector.variance, results_queue))


def perform_nms(confidence_threshold: float = 0.7, nms_threshold: float = 0.4):
    while True:
        frame_id, frame, loc, conf, land, transformed_shape, variance, results_queue = nms_queue.get()

        original_height, original_width = frame.shape[:2]
        transformed_height, transformed_width = transformed_shape
        transformed_size = (transformed_width, transformed_height)

        scale_landmarks = torch.from_numpy(np.tile(transformed_size, 5))
        scale_bboxes = torch.from_numpy(np.tile(transformed_size, 2))

        prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=transformed_shape,
        )

        conf = F.softmax(conf, dim=-1)

        annotations: List[Dict[str, Union[List, float]]] = []

        boxes = decode(loc.data, prior_box, variance)

        boxes *= scale_bboxes
        scores = conf[:, 1]

        landmarks = decode_landm(land.data, prior_box, variance)
        landmarks *= scale_landmarks

        # ignore low scores
        valid_index = torch.where(scores > confidence_threshold)[0]
        boxes = boxes[valid_index]
        landmarks = landmarks[valid_index]
        scores = scores[valid_index]

        # do NMS
        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep, :]

        if boxes.shape[0] == 0:
            results_queue.put((frame_id, [{"bbox": [], "score": -1, "landmarks": []}]))
            continue

        landmarks = landmarks[keep]

        scores = scores[keep].cpu().numpy().astype(float)

        boxes_np = boxes.cpu().numpy()

        landmarks_np = landmarks.cpu().numpy()
        resize_coeff = original_height / transformed_height

        boxes_np *= resize_coeff
        landmarks_np = landmarks_np.reshape(-1, 10) * resize_coeff

        for box_id, bbox in enumerate(boxes_np):
            x_min, y_min, x_max, y_max = bbox

            x_min = np.clip(x_min, 0, original_width - 1)
            x_max = np.clip(x_max, x_min + 1, original_width - 1)

            if x_min >= x_max:
                continue

            y_min = np.clip(y_min, 0, original_height - 1)
            y_max = np.clip(y_max, y_min + 1, original_height - 1)

            if y_min >= y_max:
                continue

            annotations += [
                {
                    "bbox": np.round(bbox.astype(float), ROUNDING_DIGITS).tolist(),
                    "score": np.round(scores, ROUNDING_DIGITS)[box_id],
                    "landmarks": np.round(landmarks_np[box_id].astype(float), ROUNDING_DIGITS)
                    .reshape(-1, 2)
                    .tolist(),
                }
            ]

        results_queue.put((frame_id, annotations))        
    

def worker(output_dir, save_image, no_cache, num_iters, device, frame_number, frame_size):   
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

                    faces, probs = crop_faces(frames, frame_size)
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
                    np.save(f"{iter_path}/{file_name}.npy", iter_faces)

                    if save_image:
                        for i, (face, prob) in enumerate(zip(iter_faces, iter_probs)):
                            image_path = f"{iter_path}/{file_name}"
                            create_folder(image_path)
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
    config = json.load(open("config.json"))
    create_folder(args.output)

    for f in os.listdir(args.input):
        if f.endswith(".mp4"):
            queue.put(f"{args.input}/{f}")

    video_workers = []
    for _ in range(8):
        video_worker = Process(target=worker, args=(
            args.output, 
            args.save_image, 
            args.no_cache,
            config['num_eval_iters'],
            config['device'],
            config['frame_number'],
            config['frame_size']
        ))
        video_worker.daemon = True
        video_worker.start()
        video_workers.append(video_worker)

    face_detection_worker = Process(target=detect_faces, args=(config['device'],))
    face_detection_worker.daemon = True
    face_detection_worker.start()

    npm_workers = []
    for _ in range(16):
        nms_worker = Process(target=perform_nms)
        nms_worker.daemon = True
        nms_worker.start()
        npm_workers += [nms_worker]

    queue.join()

    #for video_worker in video_workers:
    #    video_worker.stop()
    #for nms_worker in npm_workers:
    #    nms_worker.stop()
    #face_detection_worker.stop()

    #for w in workers:
    #    w.join()
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers",    type=int,   required=True, help="Number of workers")
    parser.add_argument("--input",      type=str,   required=True, help="Input directory of raw videos")
    parser.add_argument("--output",     type=str,   required=True, help="Output directory of faces")
    parser.add_argument("--save-image", action='store_true')
    parser.add_argument("--no-cache",   action='store_true')

    args = parser.parse_args()
    main(args)