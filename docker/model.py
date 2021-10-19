import torch.utils.data
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchmetrics
import albumentations as A
from utils import calc_prob
from pytorch_lightning.core.lightning import LightningModule
import wandb
from pytorchvideo.models.audio_visual_slowfast import create_audio_visual_slowfast


train_auc   = torchmetrics.AUROC(pos_label=1)
val_auc     = torchmetrics.AUROC(pos_label=1)
test_auc    = torchmetrics.AUROC(pos_label=1)


class Model(LightningModule):
    def __init__(self, total_steps, lr):
        super().__init__()
        self.model = create_audio_visual_slowfast(
            model_num_class=2,
            model_depth=18
        )
        self.best_auc = 0
        self.total_steps = total_steps
        self.lr = lr
        self.automatic_optimization = False
    
    def log_all(self, metrics):
        wandb.log(metrics, step=self.global_step)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        opt = self.optimizers()        
        opt.zero_grad()
        
        lr = [group['lr'] for group in opt.param_groups][0]
        fast = batch["video"][0]
        slow = batch["video"][1]
        audio = torch.unsqueeze(torch.unsqueeze(batch["audio"], 1), 3)
        print(fast.shape, slow.shape, audio.shape)
        logits = self.model([fast, slow, audio])
        loss = F.cross_entropy(logits, batch["label"])
        mean_loss = torch.mean(self.all_gather(loss))
        train_auc.update(calc_prob(logits), batch["label"])

        self.manual_backward(loss)
        opt.step()
        
        if self.global_rank == 0:
            self.log_all({
                "train/loss": mean_loss,
                "lr": lr
            })
    
    def training_epoch_end(self, _):
        auc = train_auc.compute()
        train_auc.reset()
        
        if self.global_rank == 0:
            self.log_all({
                "train/auc": auc
            })
            
    def validation_step(self, batch, _):            
        logits = self.model(batch["video"])
        loss = F.cross_entropy(logits, batch["label"])
        mean_loss = torch.mean(self.all_gather(loss))
        val_auc.update(calc_prob(logits), batch["label"])
        
        if self.global_rank == 0:            
            self.log_all({
               "val/loss": mean_loss
            })
    
    def validation_epoch_end(self, _):
        auc = val_auc.compute()
        val_auc.reset()
        
        sch = self.lr_schedulers()
        sch.step(auc)
        
        if self.global_rank == 0:
            self.best_auc = max(auc, self.best_auc)
            self.log_all({
                "val/auc": auc,
                "best_auc": self.best_auc,
                "epoch": self.current_epoch
            })
        self.log("val/auc", auc, sync_dist=True)
    
    def test_step(self, batch, _):
        logits = self.model(batch["video"])
        test_auc.update(calc_prob(logits), batch["label"])
    
    def test_epoch_end(self, _):
        auc = test_auc.compute()
        test_auc.reset()

        if self.global_rank == 0:
            self.log_all({
                "test/auc": auc
            })
        self.log("test/auc", auc, sync_dist=True)


    def predict_step(self, batch, _):
        return self(batch["video"])
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max",
            patience=3,
            cooldown=1,
            factor=0.3,
            verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/auc"}