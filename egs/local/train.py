import os
import yaml
import argparse
import logging
from torch import nn
import torch
from torch.utils.data import DataLoader
from osdc.utils import BinaryMeter
from online_data  import OnlineFeats
import random
import numpy as np
from osdc.models.tcn import TCN
from tqdm import tqdm  
from datetime import datetime
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="OSDC on AMI")
parser.add_argument("conf_file", type=str)
parser.add_argument("log_dir", type=str)
parser.add_argument("gpus", type=str)


logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
torch.set_float32_matmul_precision('medium')  # 添加这一行

seed = 1 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlainModel(nn.Module):
    def __init__(self, model):
        super(PlainModel, self).__init__()
        self.model = model

    def forward(self, x):
        mask = self.model(x)
        return mask

class OSDC_AMI(nn.Module):
    def __init__(self, hparams):
        super(OSDC_AMI, self).__init__()
        self.configs = hparams
        self.best_osd_f1 = 0
        self.best_epoch = 0
        if not self.configs["augmentation"]["probs"]:
            cross = nn.CrossEntropyLoss(torch.Tensor([1.74, 1.0, 11.98]).to(device), reduction="none")
        else:
            cross = nn.CrossEntropyLoss(torch.Tensor([1.0, 2.13, 6.89]).to(device), reduction="none")

        self.loss = lambda x, y: cross(x, y)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.train_vad_metrics = BinaryMeter()
        self.train_osd_metrics = BinaryMeter()
        self.val_vad_metrics = BinaryMeter()
        self.val_osd_metrics = BinaryMeter()

        self.model = PlainModel(TCN(120, 3, 1, 3, 3, 64, 128).to(device))
        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(" trainable parameters : {} M".format(num_param * 1e-6))


    def compare_labels(self, tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            raise ValueError("The input tensors must have the same shape.")

   
        result = (tensor1 == 0) & (tensor2 == 0) | (tensor1 != 0) & (tensor2 != 0)
        result = (~result).int()  
        return result.sum(dim=-1)
    
    def weight(self, labels, L, R):
        bs, n = labels.shape
        weight_list = torch.ones(bs, n, dtype=torch.float32)

        for j in range(L, n - R):
            labels_L = labels[:, max(0, j - L):j]  
            labels_R = labels[:, j:j + R] 

            min_length = min(labels_L.size(1), labels_R.size(1))
            labels_L = labels_L[:, -min_length:]  
            labels_R = labels_R[:, :min_length] 

            count = self.compare_labels(labels_L, labels_R).float()
            wi = 0.1 * torch.log(count + 1) + 1
            weight_list[:, j] = wi

        return weight_list
        
   
    def cross_entropy_wloss(self, preds, label):

        L = 20
        R = 20
        weights = self.weight(label, L, R).to(device)
        loss = weights*self.loss(preds, label)
        return loss

    def train_step(self, batch):
        feats, label, mask = batch
        feats = feats.to(device)
        label = label.to(device)
        mask = mask.to(device)

        preds = self.model(feats)
        mseloss = 0
 
        for p, m in zip(preds, mask):
            p_masked = p * m 
            mseloss += 0.25 * torch.mean(
             torch.clamp(self.mse_loss(F.log_softmax(p_masked[:, 1:], dim=1), F.log_softmax(p_masked.detach()[:, :-1], dim=1)), min=0, max=16), dim=1).mean()
        
        mseloss /= len(preds)

        wceloss = self.cross_entropy_wloss(preds, label)
        wceloss = wceloss * mask.detach()
        wceloss = wceloss.mean()
        loss = mseloss + wceloss
        preds = torch.softmax(preds, 1)

        self.train_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
        self.train_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)

        return loss

    def val_step(self, batch):
        feats, label, mask = batch
        feats = feats.to(device)
        label = label.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            preds = self.model(feats)
            wceloss = self.cross_entropy_wloss(preds, label)
            wceloss = wceloss * mask.detach()
            wceloss = wceloss.mean()
            mseloss = 0
 
            for p, m in zip(preds, mask):
                p_masked = p * m 
                mseloss += 0.25 * torch.mean(
                torch.clamp(self.mse_loss(F.log_softmax(p_masked[:, 1:], dim=1), F.log_softmax(p_masked.detach()[:, :-1], dim=1)), min=0, max=16), dim=1).mean()
            mseloss /= len(preds)
            loss = mseloss + wceloss
            preds = torch.softmax(preds, 1)
            self.val_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
            self.val_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)
        return loss


    def train_epoch(self, model, train_loader, optimizer, epoch):
        train_current_epoch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_log = (f"Training time of the {epoch}-th: {train_current_epoch_time}\n")
        logger.info(time_log)

        model.train()
        train_total_loss = 0.0
        train_average_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
        
        for batch_idx, batch in progress_bar:
      
            loss = model.train_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()         
            train_total_loss += loss.item()  
            progress_bar.set_postfix({"Loss": loss.item()})
        
        train_average_loss = train_total_loss / len(train_loader)  

        log_msg = (
                    f"Training Results - Epoch {epoch}:\n"
                    f"train batch:{len(train_loader)}\n"
                    f"Avg Loss: {train_average_loss}\n"
                    f"Precision VAD: {self.train_vad_metrics.get_precision()}\n"
                    f"Recall VAD: {self.train_vad_metrics.get_recall()}\n"
                    f"Aaccuracy VAD: {self.train_vad_metrics.get_accuracy()}\n"
                    f"FA VAD: {self.train_vad_metrics.get_FA()}\n"
                    f"Miss VAD: {self.train_vad_metrics.get_miss()}\n"
                    f"VAD f1: {self.train_vad_metrics.get_f1()}\n"
                    f"Aaccuracy OSD: {self.train_osd_metrics.get_accuracy()}\n"
                    f"Precision OSD: {self.train_osd_metrics.get_precision()}\n"
                    f"Recall OSD: {self.train_osd_metrics.get_recall()}\n"
                    f"OSD f1: {self.train_osd_metrics.get_f1()}\n"
                    )
        print(log_msg)
        logger.info(log_msg)



    def val_epoch(self, model, val_loader, epoch):
        val_current_epoch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_log = (f"The val time of the {epoch}-th: {val_current_epoch_time}\n")
        logger.info(time_log)
        model.eval()
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", leave=False)
        val_total_loss = 0.0
        val_average_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in progress_bar:
                loss = model.val_step(batch)
                val_total_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})

        val_average_loss = val_total_loss / len(val_loader) 
        log_msg = (
                    f"val Results - Epoch {epoch}:\n"
                    f"Val batch:{len(val_loader)}\n"
                    f"Avg Loss: {val_average_loss}\n"
                    f"Precision VAD: {self.val_vad_metrics.get_precision()}\n"
                    f"Recall VAD: {self.val_vad_metrics.get_recall()}\n"
                    f"Aaccuracy VAD: {self.val_vad_metrics.get_accuracy()}\n"
                    f"FA VAD: {self.val_vad_metrics.get_FA()}\n"
                    f"Miss VAD: {self.val_vad_metrics.get_miss()}\n"
                    f"VAD f1: {self.val_vad_metrics.get_f1()}\n"
                    f"Aaccuracy OSD: {self.val_osd_metrics.get_accuracy()}\n"
                    f"Precision OSD: {self.val_osd_metrics.get_precision()}\n"
                    f"Recall OSD: {self.val_osd_metrics.get_recall()}\n"
                    f"OSD f1: {self.val_osd_metrics.get_f1()}\n"
                    )
        print(log_msg)
        logger.info(log_msg)
        self.save_best_model(epoch)

    
    def reset(self):
        self.train_vad_metrics.reset()
        self.train_osd_metrics.reset()
        self.val_vad_metrics.reset()
        self.val_osd_metrics.reset()


    def save_best_model(self, epoch):
        osd_f1 = self.val_osd_metrics.get_f1().item()
        if osd_f1  > self.best_osd_f1 :
            self.best_osd_f1 = osd_f1
            self.best_epoch = epoch
            log_msg_best = (f"bset_epoch:{self.best_epoch}:\n"
                   f"best_osd_f1:{self.best_osd_f1}\n"
                   )
            print(log_msg_best)
            logger.info(log_msg_best)
            print(f'self.configs["log_dir"]:{self.configs["log_dir"]}')
            torch.save(self.model.state_dict(), os.path.join(self.configs["log_dir"], 'best_model.pth'))

def main():
    args = parser.parse_args()
    # print(f'args.conf_file:{args.conf_file}')
    with open(args.conf_file, "r") as conf:

        hparams = yaml.load(conf, Loader=yaml.FullLoader)
    # print(f'hparams:{hparams}')
    logger.info(hparams)

    model = OSDC_AMI(hparams)

    train_dataset = OnlineFeats(hparams["data"]["chime6_root"], hparams["data"]["label_train"],
                                hparams,    segment=hparams["data"]["segment"])
    val_dataset = OnlineFeats(hparams["data"]["chime6_root"], hparams["data"]["label_val"],
                              hparams, segment=hparams["data"]["segment"])
 


    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"],
                              shuffle=True, num_workers=hparams["training"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"],
                            shuffle=True, 
                            num_workers=hparams["training"]["num_workers"],
                             pin_memory=True
                             )


    print(f'val_loader:{len(val_loader)}')

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["opt"]["learning_rate"], weight_decay=hparams["opt"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

 
    for epoch in range(hparams["training"]["n_epochs"]):
    
        model.train_epoch(model, train_loader, optimizer,epoch)
        model.val_epoch(model, val_loader,epoch)
        model.save_best_model(epoch)
        scheduler.step(model.best_osd_f1)
        model.reset()

if __name__ == "__main__":
    main()
