import wandb
run = wandb.init(project="deepfake", config={"data_version": "!23123"})
print(run.name)