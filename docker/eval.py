import os
import pytorch_lightning
import torch.utils.data
import torch
import pandas as pd
import time
from utils import log, transform, predict, create_folder
from sklearn import metrics
import pandas as pd
from argparse import ArgumentParser
import numpy as np
from dataset import Dataset
import json
from pynvml import *
from model import Model


def main(input_dir, output_file):
    config = json.load(open("config.json"))
    num_gpu = 0 if config["device"] == "cpu" else torch.cuda.device_count()

    # read the video files
    test_videos = [video for video in sorted(os.listdir(input_dir)) if ".mp4" in video]
    test_data = pd.DataFrame({
        "filename": test_videos,
        "weight": [1] * len(test_videos),
        "filepath": ["metadata"] * len(test_videos),
        "label": [1] * len(test_videos)
    })
    test_data.to_csv("test.csv", index=False)    

    # crop the faces from the videos 
    log("cropping videos")
    create_folder("metadata")   
    tik = time.time()
    os.system(
        f"python crop.py "
        f"--workers {2} "
        f"--input {input_dir} "
        f"--output metadata "
    )
    avg_crop_time = (time.time() - tik) / len(test_videos)
    log(f'Face cropping completed! Average crop time: {avg_crop_time} for {len(test_videos)} videos.')
    time.sleep(10)
    
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
        gpus=num_gpu,
        precision=32 if config["device"] == "cpu" else 16,
    )

    # perform predictions
    test_dataset = Dataset(
        pd.read_csv("test.csv"),
        transform,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["submit_batch_size"],
        sampler=torch.utils.data.SequentialSampler(test_dataset),
        num_workers=0,
    )

    probs = predict(trainer, model, test_dataloader)
    probs = np.array([res["prob"] for res in probs])

    # create output and write the ouput as csv
    result = pd.DataFrame({"filename": test_videos, "probability": probs})

    label_file = f"{input_dir}/label.csv"
    if os.path.exists(label_file):
        label = pd.read_csv(label_file)
        if set(label["filename"]) == set(result["filename"]):
            merged = pd.merge(result, label, on="filename")
            fpr, tpr, _ = metrics.roc_curve(
                merged.label, 
                merged.probability, 
                pos_label=1
            )
            auc = metrics.auc(fpr, tpr)
            log("test auc:", auc)

    result.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-input",  type=str, required=True, help="Input directory of test videos")
    parser.add_argument("-output", type=str, required=True, help="Output directory with filename e.g. /data/output/submission.csv")
    args = parser.parse_args()

    main(input_dir=args.input, output_file=args.output)