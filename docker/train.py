import pytorch_lightning
import torch.utils.data
import torch
import time
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from utils import log, set_seed, calc_prob, predict
import numpy as np
import wandb
from data_loader import DataLoader
from model import Model
import json


def main(args):
    # read configuration
    config = json.load(open("config.json"))
    set_seed(config["seed"])
    num_gpu = 0 if config["device"] == "cpu" else torch.cuda.device_count()
    
    wandb_config = {
        "data_version":     args.data_version,
        "code_version":     args.code_version,
        "pret_version":     args.pret_version,
        "num_gpu":          num_gpu,
        "batch_size":       config["batch_size"] * num_gpu,
        "image_size":       config["image_size"],
        "max_lr":           config["lr"]
    }
    run = wandb.init(project="deepfake", config=wandb_config)
    
    # set up the data
    data_module = DataLoader(
        args.data_version,
        config["root_path"],
        config["frame_number"], 
        config["frame_size"],
        config["batch_size"]
    )        

    # create the model
    total_steps = config["epoch"] * int(np.ceil(data_module.get_trainset_size() / config["batch_size"]))
    model = Model(
        total_steps=total_steps,
        lr=config['lr']
    )

    # load the pretrain weight
    pretrain = torch.load(f"{config['root_path']}/pretrain/{args.pret_version}/model.ckpt")
    if "state_dict" in pretrain:
        pretrain = pretrain["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(pretrain)
    log("missing_keys   :\t", missing_keys)
    log("unexpected_keys:\t", unexpected_keys)
    
    # setup the trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val/auc",
        dirpath=f"{config['root_path']}/output/{run.id}",
        filename="model",
        save_top_k=1,
        mode="max",
    )

    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=0,
        max_epochs=config["epoch"],
        gpus=num_gpu,
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs=1,
        precision=32 if config["device"] == "cpu" else 16,
    )
    
    # start training and validating
    trainer.fit(model, data_module)
    #time.sleep(10)

    # perform evaluation on the test set
    set_seed(config["seed"])
    best_path = f"{config['root_path']}/output/{run.id}/model.ckpt"
    state_dict = torch.load(best_path)["state_dict"]
    model.load_state_dict(state_dict)

    trainer.test(model, [data_module.test_dataloader()])
    val_probs  = predict(trainer, model, data_module.val_dataloader()).cpu().detach().numpy()
    test_probs = predict(trainer, model, data_module.test_dataloader()).cpu().detach().numpy()

    #val_logits = pd.read_csv([{"filename": example, "prob": prob} \
    #    for prob, example in zip(val_probs, data_module.val_dataloader())])
    print(data_module.val_dataloader().dataset)
    #val_logits.to_csv("val_logits.csv", index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data-version', type=str)
    parser.add_argument('--code-version', type=str)
    parser.add_argument('--pret-version', type=str)
    args = parser.parse_args()
    main(args)
