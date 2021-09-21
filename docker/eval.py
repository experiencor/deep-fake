import os
import pytorch_lightning
import torch.utils.data
import torch
import time
from utils import log, calc_prob, set_seed, transform, compute_num_crop_workers
import pandas as pd
import logging
from argparse import ArgumentParser
import numpy as np
from dataset import Dataset
import json
from pynvml import *
from model import Model


def main(input_dir, output_file):
    config = json.load(open("config.json"))
    if not os.path.exists("/data/faces"):
        os.mkdir("/data/faces")
    num_gpu = 0 if config["device"] == "cpu" else torch.cuda.device_count()

    # read the video files and make the test file
    test_videos = [video for video in os.listdir(input_dir) if ".mp4" in video]
    test_data = pd.DataFrame({
        "filename": test_videos,
        "label": [1] * len(test_videos)
    })
    test_data.to_csv("test.csv", index=False)

    # crop the faces from the videos    
    log("cropping videos")
    tik = time.time()
    num_crop_workers = compute_num_crop_workers()
    log(f"num_crop_workers: {num_crop_workers}")
    os.system(
        f"python crop.py --workers {num_crop_workers} --num-iters {config['num_eval_iters']} "
        f"--input {input_dir} --output /data/faces")
    avg_crop_time = (time.time() - tik) / len(test_videos)
    logging.warning(
        f'Face cropping completed! Average crop time: '
        f'{avg_crop_time} for {len(test_videos)} videos.')
    time.sleep(15)

    face_count = 0
    for face in os.listdir("/data/faces/0/"):
        if ".npy" in face:
            face_count += 1
    if face_count != len(test_videos):
        return
    
    # create the model and load the weights
    model = Model(0, 0)
    
    if num_gpu == 0:
        pretrain = torch.load("model.ckpt", map_location=torch.device('cpu'))
    else:
        pretrain = torch.load("model.ckpt")
        
    if "state_dict" in pretrain:
        pretrain = pretrain["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(pretrain)
    log("missing_keys   :\t", missing_keys)
    log("unexpected_keys:\t", unexpected_keys)
    
    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=0,
        gpus=num_gpu
    )

    # perform predictions
    set_seed(config["seed"])
    final_results = []
    for iteration in range(1):
        test_dataset = Dataset(
            data_frame="test.csv",
            video_path_prefix=f"/data/faces/{iteration}",
            frame_number=config['frame_number'],
            frame_size=config['frame_size'],
            transform=transform
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            sampler=torch.utils.data.SequentialSampler(test_dataset),
            num_workers=4,
            prefetch_factor=2,
        )

        logits = trainer.predict(model, test_dataloader)
        logits = torch.cat(logits)
        probs  = calc_prob(logits)
        final_results += [probs]
    final_results = torch.mean(torch.stack(final_results), dim=0).cpu().detach().numpy()
    
    # create output and write the ouput as csv
    output_df = pd.DataFrame({"filename": test_videos, "probability": final_results})
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-input",  type=str, required=True, help="Input directory of test videos")
    parser.add_argument("-output", type=str, required=True, help="Output directory with filename e.g. /data/output/submission.csv")
    args = parser.parse_args()

    main(input_dir=args.input, output_file=args.output)