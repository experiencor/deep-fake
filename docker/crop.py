import cv2
import os
import matplotlib.pylab as plt
from multiprocessing import Process, JoinableQueue
import traceback
import json
import warnings
import time
from retinaface.pre_trained_models import get_model
import numpy as np
from SyncNetModel import *
import librosa.display
import python_speech_features
from argparse import ArgumentParser
from utils import log, create_folder, calc_pdist
from scipy.ndimage.filters import uniform_filter1d
import time
import librosa
import torch
from moviepy.editor import AudioFileClip, VideoFileClip, CompositeAudioClip
from moviepy.video.io import ImageSequenceClip
import torch        
import torchaudio.transforms as T        

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
queue = JoinableQueue()
extraction_queue = JoinableQueue()
fail_queue = JoinableQueue()


def apply_crop(frame, size, y, x, margin=0.4):
    bs = size
    bsi = int(bs*(1+2*margin))  # Pad videos by this amount 

    frame = np.pad(frame,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = y+bsi  # BBox center Y
    mx  = x+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*margin)),int(mx-bs*(1+margin)):int(mx+bs*(1+margin))]
    return face


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

        return_boxes += [boxes[0]]
        return_probs += [probs[0]]
    return return_boxes, return_probs

def compute_latency(__S__, faces, audio, resample_rate, batch_size=8, vshift=15):
    with torch.no_grad():
        lastframe = len(faces) - 5
        faces = np.stack(faces, axis=3)
        faces = np.expand_dims(faces, axis=0)
        faces = np.transpose(faces, (0,3,4,1,2))

        mfcc = zip(*python_speech_features.mfcc(audio, resample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        im_feat = []
        cc_feat = []

        for i in range(0, lastframe, batch_size):
            im_batch = [ torch.tensor(faces[:, :, vframe:vframe+5, :, :]) for vframe in range(i, min(lastframe, i+batch_size)) ]
            im_in = torch.cat(im_batch, 0)
            im_out  = __S__.forward_lip(im_in.cuda().half())
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:, :, :, vframe*4:vframe*4+20] for vframe in range(i, min(lastframe, i+batch_size)) ]
            cc_in = torch.cat(cc_batch, 0)
            cc_out  = __S__.forward_aud(cc_in.cuda().half())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1).float()
        
        minval, minidx = torch.min(mdist, 0)
        offset = vshift-minidx
        conf   = torch.median(mdist) - minval

        return mdist, offset, conf

def extract(save_image, frame_num, frame_size, output):
    while True:
        log(">" * 60, "extraction_queue", extraction_queue.qsize())
        elem = extraction_queue.get()

        if elem == None:
            extraction_queue.task_done()
            return

        file_path, start, end, select_faces, mdist, latency, conf = elem
        file_name = file_path.split("/")[-1].split(".")[0]
        
        # extract audio frames
        try:
            offset = 0
            duration = (end - start) / frame_num
            mels = []
            audioclip = AudioFileClip(file_path)
            audioclip.write_audiofile(f"{output}/{file_name}.wav", 44100, 2, 2000, "pcm_s32le")
            for i in range(frame_num):
                samples,sample_rate = librosa.load(f"{output}/{file_name}.wav", offset=offset, duration=duration)
                mel = librosa.power_to_db(librosa.feature.melspectrogram(y=samples, sr=sample_rate))
                mel = cv2.resize(mel, (frame_size, frame_size))
                mels += [mel]
                offset += duration
            os.system(f"rm {output}/{file_name}.wav")
        except Exception as e:
            log("No audio found!", e)
            mels = []

        mel_3cs = []
        if save_image:
            visible_folder = f"{output}/{file_name}"
            create_folder(visible_folder)
            for i, face in enumerate(select_faces):
                cv2.imwrite(f"{visible_folder}/img_{'%05d' % (1+i)}.jpg", face)

            if len(mels) > 0:
                for i, mel in enumerate(mels):
                    fig = plt.figure(figsize=[1,1])
                    ax =fig.add_subplot(111)
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    ax.set_frame_on(False)

                    librosa.display.specshow(mel)

                    plt.savefig(
                        f"{visible_folder}/img_{'%05d' % (1+i+frame_num)}.jpg", 
                        dpi=1000, 
                        bbox_inches="tight",
                        pad_inches=0
                    )
                    plt.close('all')

                    mel_3cs += [cv2.imread(f"{visible_folder}/img_{'%05d' % (1+i+frame_num)}.jpg")]

        # save input as numpy array
        np.savez_compressed(f"{output}/{file_name}", 
            faces=select_faces, 
            mel_3cs=mel_3cs, mdist=mdist, latency=latency, conf=conf)

        extraction_queue.task_done()

def work(
    save_avi,
    resample_rate,
    device, 
    max_clip_len,
    skip_num,
    frame_size, 
    frame_num,
    start_time,
    video_num,
    output
):    
    try:
        face_detector = get_model(
            model_name="resnet50_2020-07-20",
            max_size=2048, 
            device=device
        )
        face_detector.eval()

        __S__ = S(num_layers_in_fc_layers = 1024).cuda().half().eval()
        loaded_state = torch.load("syncnet_v2.model", map_location=lambda storage, loc: storage)
        state = __S__.state_dict()
        for name, param in loaded_state.items():
            state[name].copy_(param)
    except Exception as e:
        log(e)
        return

    while True:
        log(">" * 60, f"file_path queue: {queue.qsize()}, "
            f"average crop time: {round((time.time() - start_time) / (video_num - queue.qsize() + 1e-3), 2)}")
        file_path = queue.get()
        
        if file_path is None:
            queue.task_done()
            return

        if fail_queue.qsize() > 8:
            queue.task_done()
            continue

        file_name = file_path.split("/")[-1].split(".")[0]

        try:
            tik = time.time()
            videoclip = VideoFileClip(file_path)
            start, end = 0, min(max_clip_len, videoclip.duration)

            frames = []
            for frame in videoclip.subclip(start, end).iter_frames(fps=25):
                frames += [frame]
            frames = np.array(frames)

            # make and smooth bboxes
            all_boxes, all_probs = crop_faces(file_path, face_detector, frames, skip_num)

            all_sizes = [max((box[3]-box[1]), (box[2]-box[0]))/2 for box in all_boxes]
            all_ys = [(box[1]+box[3])/2 for box in all_boxes]
            all_xs = [(box[0]+box[2])/2 for box in all_boxes]

            all_ys = [y + np.random.randint(-y*0.015, y*0.015+1) for y in all_ys]
            all_xs = [x + np.random.randint(-x*0.015, x*0.015+1) for x in all_xs]
            
            all_sizes = uniform_filter1d(all_sizes, size=13)
            all_ys = uniform_filter1d(all_ys, size=13)
            all_xs = uniform_filter1d(all_xs, size=13)

            # apply the bboxes to the frames
            faces, probs = [], []
            for frame, size, y, x, prob in zip(frames, all_sizes, all_ys, all_xs, all_probs):
                face = apply_crop(frame, size, y, x)
                faces += [cv2.resize(face, (frame_size, frame_size))[:,:,::-1]]
                probs += [prob]
            log(f"video processing time: {time.time() - tik}")
            tik = time.time()

            # read audio and resample audio
            audioclip = AudioFileClip(file_path).subclip(start, end)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = audioclip.to_soundarray()[:, 0]
            audio_rate = int(len(audio)/(end - start))
            audio = torch.tensor(librosa.resample(
                audio,
                audio_rate,
                resample_rate
            )).float()

            # save the avi and compute the latency
            if save_avi:
                videoclip = ImageSequenceClip.ImageSequenceClip(faces, fps=25)
                new_audioclip = CompositeAudioClip([audioclip])
                videoclip.audio = new_audioclip
                videoclip.write_videofile(f"{output}/{file_name}.avi", codec="rawvideo")
            mdist, offset, conf = compute_latency(__S__, faces, audio, resample_rate, batch_size=8, vshift=15)
            log(f"audio processing time: {time.time() - tik}")
            log(f"offset of {file_name}: {offset}")

            # uniform sampling of frames
            step = len(faces)//frame_num
            indices = [i * step for i in range(frame_num)]
            select_faces = [faces[i] for i in indices]

            extraction_queue.put((
                file_path,
                start,
                end, 
                select_faces, 
                mdist,
                offset,
                conf
            ))
        except Exception as e:
            queue.put(file_path)
            fail_queue.put((file_path))
            log("=" * 20)
            log(e)
            log(f"number of failed videos: {fail_queue.qsize()}")
            traceback.print_exc()
        queue.task_done()
        

def main(args):
    tik = time.time()
    config = json.load(open("config.json"))
    create_folder(args.output)

    files = [f for f in sorted(os.listdir(args.input)) if f.endswith(".mp4")]
    for f in files:
        queue.put(f"{args.input}/{f}")

    # start the face detectors
    workers = []
    for _ in range(args.workers):
        worker = Process(target=work, args=(     
            args.save_avi,       
            config['resample_rate'],
            config['device'],
            config['max_clip_len'],
            config['face_detection_step'],
            config['frame_size'],
            config['frame_num'],
            tik,
            len(files),
            args.output, 
        ))
        worker.start()
        workers.append(worker)

    # start the audio processors
    extractors = []
    for i in range(args.workers):
        extractor = Process(target=extract, args=(
            args.save_image,            
            config['frame_num'],
            config['frame_size'],
            args.output,            
        ))
        extractor.start()
        extractors += [extractor]

    # wait for the queues to be empty
    queue.join()
    extraction_queue.join()

    # put Nones tp stop the processes
    for _ in range(args.workers):
        queue.put(None)
    for _ in range(args.workers):
        extraction_queue.put(None)

    # wait for the workers to finish
    for worker in workers:
        worker.join()
    for extractor in extractors:
        extractor.join()

    log(f'average crop time: {(time.time() - tik) / len(files)}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers",    type=int,   required=True, help="Number of workers")
    parser.add_argument("--input",      type=str,   required=True, help="Input directory of raw videos")
    parser.add_argument("--output",     type=str,   required=True, help="Output directory")
    parser.add_argument("--save-image", action='store_true')
    parser.add_argument("--save-avi",   action='store_true')

    args = parser.parse_args()
    main(args)