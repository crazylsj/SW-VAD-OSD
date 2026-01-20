import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from osdc.models.tcn_sacc import TCN
# local/osdc/models/tcn_channelattGCT.py /mnt/sdb1/lsj/OSDC-mamba/egs/local/osdc/models/tcn_channelattGCT.py
# from osdc.model.tcn_channelattGCT import TCN
# from osdc.models.tcn_complict import TCN
from osdc.models.case.case_zq import TCN
# from osdc.models.caseA.case3 import TCN
# from osdc.models.tcn_channelattGCT import TCN

# from osdc.models.mamba import ModelArgs
# from online_data_array import OnlineFeats
# from online_data_test import OnlineFeats_test
from online_data_8channel import OnlineFeats
# from online_data_csd8_array import OnlineFeats
# from online_data_array import OnlineFeats
from osdc.utils import BinaryMeter, MultiMeter
import yaml
import os
import random
from torch import nn
from tqdm import tqdm  # 用于显示训练进度条
import numpy as np
from datetime import datetime
from thop import profile

seed = 1 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlainModel(nn.Module):
    def __init__(self, masker):
        super(PlainModel, self).__init__()
        self.model = masker

    def forward(self, tf_rep):
        mask = self.model(tf_rep)
        return mask

class OSDC_AMI(pl.LightningModule):
    def __init__(self, hparams,test_loader,model_checkpoint):
        super(OSDC_AMI, self).__init__()
        self.configs = hparams
        self.test_loader = test_loader
        # self.best_osd_f1 = 0
        # self.best_osd_ap = 0
        # self.best_epoch = 0


        self.test_batch_ap_vad_sum = 0
        self.test_batch_ap_osd_sum = 0
        self.test_batch_count = 0

        self.count = 0 

   

        self.test_ap_vad = 0
        self.test_ap_osd = 0

        # if not self.configs["augmentation"]["probs"]:
        #     cross = nn.CrossEntropyLoss(torch.Tensor([1.74, 1.0, 11.98, 219, 1000]).to(device), reduction="none")
        # else:
        #     cross = nn.CrossEntropyLoss(torch.Tensor([1.0, 2.13, 6.89, 20, 115]).to(device), reduction="none")
        if not self.configs["augmentation"]["probs"]:
            cross = nn.CrossEntropyLoss(torch.Tensor([1.74, 1.0, 11.98]).to(device), reduction="none")
        else:
            cross = nn.CrossEntropyLoss(torch.Tensor([1.0, 2.13, 6.89]).to(device), reduction="none")

        self.loss = lambda x, y: cross(x, y)

        self.test_vad_metrics = BinaryMeter()
        self.test_osd_metrics = BinaryMeter()

        self.model = PlainModel(TCN(64, 3, 1, 3, 3, 64, 128).to(device))
        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # from thop import profile
        # from thop import clever_format
        # input = torch.randn(1, 8, 80000)
        # flops, params = profile(self.model, inputs=(input, ),verbose=False)

        # # 格式化 FLOPs 和参数量的输出
        # flops, params = clever_format([flops, params], "%.3f")
        # print(f"FLOPs: {flops}, Params: {params}")
        print("singlechannel array1 Number of trainable parameters : {} M".format(num_param * 1e-6))


        self.checkpoint = torch.load(model_checkpoint)
        self.model.load_state_dict(self.checkpoint)


    def test_step(self, batch):
        feats, label, mask = batch
        feats = feats.to(device)
        label = label.to(device)
        mask = mask.to(device)
        # print(f'feats:{feats.shape}')
       

        with torch.no_grad():
            # print(f'self.count"{self.count}')
            self.count += 1
            preds = self.model(feats)
    
            # print(f'preds1:{label.shape}')
            loss = self.loss(preds, label)
            loss = loss * mask.detach()
            loss = loss.mean()
            preds = torch.softmax(preds, 1)
 
            # print(f'VAD指标计算')
            self.test_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
            # print(f'OSD指标计算')
            self.test_osd_metrics.update(torch.sum(preds[:, 2:], 1), label == 2)

        return loss

    def test_epoch(self, model):
       
        model.eval()
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="test", leave=False)
        test_total_loss = 0.0
        test_average_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in progress_bar:
                loss = model.test_step(batch)
                  
                test_total_loss += loss.item()  # 累加损失
        
                progress_bar.set_postfix({"Loss": loss.item()})

        test_average_loss = test_total_loss / len(self.test_loader)  # 计算平均损失
        # self.test_ap_vad = self.test_batch_ap_vad_sum / self.test_batch_count
        # self.test_ap_osd = self.test_batch_ap_osd_sum / self.test_batch_count


        log_msg = (
                        f"测试总batch:{len(self.test_loader)}\n"
                        f"Avg Loss: {test_average_loss}\n"
                        f"MR VAD: {self.test_vad_metrics.get_MR()}\n"
                        f"FAR VAD: {self.test_vad_metrics.get_FAR()}\n"
                        f"HTER VAD: {self.test_vad_metrics.get_HTER()}\n"
                        f"Precision OSD: {self.test_osd_metrics.get_precision()}\n"
                        f"Recall OSD: {self.test_osd_metrics.get_recall()}\n"
                        f"OSD f1: {self.test_osd_metrics.get_f1()}\n"
                        )
        print(log_msg)


def main(conf_file, log_dir, model_checkpoint, test_data):
    # 加载配置文件
    with open(conf_file, 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    print(f'hparams: {hparams}')
    
    # 设置测试数据加载器
    dataset = OnlineFeats_test(hparams["data"]["chime6_root"], hparams["data"]["label_test"],
                            hparams, segment=hparams["data"]["segment"])
    test_loader = DataLoader(dataset, batch_size=hparams["training"]["batch_size"],
                                shuffle=False,  # 关闭 shuffle
                                num_workers=hparams["training"]["num_workers"], drop_last=False)

    # 加载模型
    model = OSDC_AMI(hparams,test_loader,model_checkpoint)

    model.test_epoch(model)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OSDC on AMI")
    parser.add_argument("conf_file", type=str, help="Path to the configuration file")
    parser.add_argument("log_dir", type=str, help="Directory to save logs")
    parser.add_argument("model_checkpoint", type=str, help="Path to the model checkpoint file")
    parser.add_argument("test_data", type=str, help="Path to the test data directory")
    
    args = parser.parse_args()
    main(args.conf_file, args.log_dir, args.model_checkpoint, args.test_data)
