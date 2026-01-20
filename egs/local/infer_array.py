import torch
import os
import argparse
import glob
# from glob import glob
import soundfile as sf
from torchaudio.compliance.kaldi import mfcc
from osdc.utils.oladd import overlap_add
import numpy as np
# from osdc.features.ola_feats import compute_feats_windowed_array
from osdc.features.ola_feats import compute_feats_windowed_diagonal_csd, compute_feats_windowed_frequency_selected_diagonal_csd, compute_feats_windowed_array_csd8_topmax10,compute_feats_windowed_csd8_array
import yaml
# from train import OSDC_AMI
from train_csdop_tcn_ce_mseloss import OSDC_AMI
from osdc.models.tcn import TCN
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser("Single-Channel inference, average logits")
parser.add_argument("exp_dir", type=str)
# parser.add_argument("checkpoint_name", type=str)
parser.add_argument("wav_dir", type=str)
parser.add_argument("out_dir", type=str)
parser.add_argument("gpus", type=str, default="0")
parser.add_argument("--window_size", type=int, default=498)
parser.add_argument("--lookahead", type=int, default=0)
parser.add_argument("--lookbehind", type=int, default=0)
parser.add_argument("--regex", type=str, default="")

class PlainModel(nn.Module):
    def __init__(self, masker):
        super(PlainModel, self).__init__()
        self.model = masker

    def forward(self, tf_rep):
        mask = self.model(tf_rep)
        return mask


def plain_single_file_predict(args, model, wav_dir, train_configs, out_dir, window_size=400, lookahead=200, lookbehind=200, regex=""):
    checkpoint = torch.load('/home/lsj/OSDC-Loss/egs/exp/tcn/best_model_training_MA_csd8_tcn_ce.pth')
    model.load_state_dict(checkpoint)
   

    model = model.eval().cuda()
    # print(f'{os.path.join(wav_dir, "**/*.wav").format(regex)}')
    audio_files = glob.glob(os.path.join(wav_dir, "**/*.Array1-*.wav").format(regex), recursive=True)
    wavs = []
    audio_s = []
    for audio in audio_files:
        audio_s.append(audio)
        if len(audio_s) == 8:
            wavs.append(sorted(audio_s))
            audio_s = []
    # print(f'audio_sess:{audio_sess}')


        

    # wavs = glob(os.path.join(wav_dir, "**/*{}*.wav"), recursive=True)
    # print(f'wavs:{audio_files}')

    # assert len(wavs) > 0, "No file found"

    for wav in wavs:
        print(f'wav:{wav}')
        print("Processing File {}".format(wav))
        audio, _ = sf.read(wav[0])
        # print(f'audio:{audio.shape}')
        # print(f'train_configs["feats"]["type"] :{train_configs["feats"]["type"] }')
        # print(f'**train_configs["mfcc_kaldi"]:{train_configs["mfcc_kaldi"]}')


        if train_configs["feats"]["type"] == "mfcc_kaldi":
            feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                             **train_configs["mfcc_kaldi"]).transpose(0, 1)
        else:
            raise NotImplementedError
        # print(f'feats_func:{feats_func.shape}')
        tot_feats = compute_feats_windowed_array_csd8_topmax10(feats_func, audio, wav)
        print(f'tot_feats:{tot_feats.shape}')
        tot_feats = tot_feats.detach().cpu().numpy()
        pred_func = lambda x : model(torch.from_numpy(x).unsqueeze(0).cuda()).detach().cpu().numpy()
        # print(f'pred_func:{pred_func.shape}')
        preds = overlap_add(tot_feats, pred_func, window_size, window_size // 2, lookahead=lookahead, lookbehind=lookbehind)
        print(f'preds:{preds.shape}')
        out_file = os.path.join(out_dir, wav[0].split("/")[-1].split(".wav")[0] + ".logits")
        np.save(out_file, preds)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(args.exp_dir, "confs.yml"), "r") as f:
        confs = yaml.load(f, Loader=yaml.FullLoader)
    # test if compatible with lightning
    confs.update(args.__dict__)

    model = PlainModel(TCN(440, 3, 1, 3, 3, 64, 128).to(device))
  

    # model.load_state_dict(state_dict)
    # model = model.model
    os.makedirs(confs["out_dir"], exist_ok=True)
    plain_single_file_predict(args,model, confs["wav_dir"],
                              confs, confs["out_dir"], window_size=args.window_size,
                              lookahead=args.lookahead, lookbehind=args.lookbehind, regex=args.regex)
