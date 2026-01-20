from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from osdc.utils.oladd import _gen_frame_indices
import random
from pysndfx import AudioEffectsChain
from scipy.signal import csd, windows
import soundfile as sf
import traceback
seed = 1 
random.seed(seed)
# 固定 NumPy 的随机数生成器
np.random.seed(seed)
# 固定 PyTorch 的随机数生成器
torch.manual_seed(seed)
# 如果使用了 GPU，加上这行固定 CUDA 的随机数生成器
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# 确保使用确定性算法
torch.backends.cudnn.deterministic = True
# 禁用 cuDNN 的自动优化特性，确保结果一致
torch.backends.cudnn.benchmark = False
# import librosa
class MultichannelSTFT(torch.nn.Module):
    """
    Compute STFT on eah channel of a multichannel audio signal and stack them

    :param in_channels: number of channels (i.e. microphones) in the input signal
    :param n_fft: number of frequency bins in the stft (default: 512)
    :param win_length: length of the stft window (default: 400)
    :param hop_length: hop of the stft window (default: 160)
    :param center: wether the window position is defined from the center or the start (default: True)
    :param pad: padding applied on both sides of the input signal (default: 0)
    """

    def __init__(
        self,
        in_channels = 8,
        n_fft=400,
        win_length=400,
        hop_length=160,
        center=False,
        pad=0,
    ):
        super(MultichannelSTFT, self).__init__()

        self.stft_kw = dict(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            return_complex=True,
        )
        self.in_channels = in_channels
        self.pad = pad

    def forward(self, x):
        x_padded = F.pad(x, (self.pad, self.pad))
        X = [
            torch.stft(x_padded[i, :], **self.stft_kw)
            for i in range(self.in_channels)
        ]
        # print(f'torch.stack(X):{torch.stack(X).shape}')
        return torch.stack(X).permute(1, 0, 2)  # (freq channel, time)
    
class OnlineFeats_test(Dataset):

    def __init__(self, ami_audio_root, label_root, configs, segment=500, probs=None, synth=None):

        self.configs = configs
        self.segment = segment
        self.probs = probs
        self.synth = None

        audio_files = glob.glob(os.path.join(ami_audio_root, "**/*.wav"), recursive=True)
        # print(f'audio_files:{audio_files}')
        for f in audio_files:
            if len(sf.SoundFile(f)) < self.segment:
                print("Dropping file {}".format(f))
        self.labels = glob.glob(os.path.join(label_root, "*.wav"))
        self.stft = MultichannelSTFT()

        # print(labels)
        # print(len(labels))
        lab_hash = {}

        for l in self.labels:
            # print(f'l:{l}')
            # l :/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/train/LABEL-EN2006a.wav
            # l_sess:EN2006a
            l_sess = str(Path(l).stem).split("-")[-1]
            # print(f'l_sess:{l_sess}')
            # print(f'l :{l }')
            lab_hash[l_sess] = l

        devices_hash = {}
        devices = []
        for f in audio_files:
            sess = Path(f).stem.split(".")[0] #sess:IS1008d 在label的范围中
           
            if sess not in lab_hash.keys():
                # print(f'sess:{sess}')
                # print("Skip session because we have no labels for it")
                continue
            devices.append(f)
            
           
            if sess not in devices_hash.keys():
                devices_hash[sess] = [f]
                # print(f'sess:{sess}')
                # print(f'f:{f}')
            else:
                devices_hash[sess].append(f)
        # print(f'-------------------------------------------devices:{devices}---------------------------------------------')
        # print(f'-------------------------------------------devices_hash:{devices_hash}-----------------------------------')


        self.devices = devices  #----保存的所有音频路径
        self.devices_hash = devices_hash # used for data augmentation ----保存所有的音频路径。但是格式为{‘IS1008c:[IS1008c.Array1-01.wav,IS1008c.Array1-02.wav,...,]’}
        # print(f'self.devices_hash:{self.devices_hash}')
        #assert len(set(list(meta.keys())).difference(set(list(lab_hash.keys())))) == 0
        # remove keys

        self.label_hash = lab_hash #-保存所有的标签路径，格式：{‘IS1008b’:''/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/dev/LABEL-IS1008b.wav'}
       
        self.tot_length = int(np.sum([len(sf.SoundFile(l)) for l in self.labels]) / segment)
        

        self.set_feats_func()

        if synth:
            self.synth=synth
            # using synthetic data.

    def get_segs(self, label_vector, min_speakers,  max_speakers):

        segs = []
        label_vector =  np.logical_and(label_vector <= max_speakers, label_vector >= min_speakers)
        changePoints = np.where((label_vector[:-1] != label_vector[1:]) == True)[0]
        changePoints = np.concatenate((np.array(0).reshape(1, ), changePoints))

        if label_vector[0] == 1:
            start = 0
        else:
            start = 1
        for i in range(start, len(changePoints) - 1, 2):
            if (changePoints[i + 1] - changePoints[i]) > 30: # if only more than 30 frames
                segs.append([changePoints[i] +1, changePoints[i + 1]-1])

        return segs

    def set_feats_func(self):

        # initialize feats_function
        if self.configs["feats"]["type"] == "mfcc_kaldi":
            from torchaudio.compliance.kaldi import mfcc
            self.feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)), **self.configs["mfcc_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "fbank_kaldi":
            from torchaudio.compliance.kaldi import fbank
            self.feats_func = lambda x: fbank(torch.from_numpy(x.astype("float32").reshape(1, -1)), **self.configs["fbank_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "spectrogram_kaldi":
            from torchaudio.compliance.kaldi import spectrogram
            self.feats_func = lambda x: spectrogram(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                             **self.configs["spectrogram_kaldi"]).transpose(0, 1)
        else:
            raise NotImplementedError

    def __len__(self):
        # return self.tot_length
        return len(self.labels)

    @staticmethod
    def normalize(signal, target_dB):

        fx = (AudioEffectsChain().custom(
            "norm {}".format(target_dB)))
        signal = fx(signal)
        return signal

    def __getitem__(self, item):
        # print(f'self.tot_length:{self.tot_length}')
        # print(item)
        sess = str(Path(self.labels[item]).stem).split("-")[-1]
        # print(f'sess:{sess}')

        
        label_path = self.labels[item]
        print(f'label_path:{label_path}')
        # print(f'label_path:{label_path}')

        
        # print(f'label_path:{type(label_path)}')
        with sf.SoundFile(label_path) as f:
            file_length = len(f)
        # print(f'file_length:{file_length}')
            
        
        start = 0
        stop = file_length
        # if label_path=='/mnt/sdb1/lsj/datasets/AMI/media/sam/bx500/amicorpus/fa_labels/eval/LABEL-TS3003b.wav':
        #     print(f'label_path:{label_path}')
        #     print(f'file_length:{file_length}')
        #     print(f'start:{start}')
        #     print(f'stop:{stop}')
        label, _ = sf.read(label_path, start=start, stop=stop)
        # print(f'labe:{label}')
        # print(f'label:{label.shape}')
        
        if self.configs["task"] == "vad":
            label = label >= 1
        elif self.configs["task"] == "osd":
            label = label >= 2
        elif self.configs["task"] == "vadosd":
            label = np.clip(label, 0, 2)
        elif self.configs["task"] == "count":
            pass
        else:
            raise EnvironmentError
        # print(self.configs["data"]["fs"])
        # print(self.configs["feats"]["hop_size"])
        start = int(start * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
        stop = int(stop * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
        # print(f'start:{start}')
        # print(f'stop:{stop}')
        # ----------------------------------所有方法都要用的---------------------
        # print(f'labe:{label}')
        
        audio_list = self.devices_hash[sess]
        # print(f'sess:{sess}')
        # print(f'audio_list:{audio_list}')
        array1_audio_list = sorted([audio_path for audio_path in audio_list if 'Array1' in audio_path])
        concatenated_audio = []
        for audio_file in array1_audio_list:
            try:
                audio, fs = sf.read(audio_file, start=start, stop=stop)
                # print(f'audio:{audio.shape}')
                if audio.ndim > 1:
                    audio = audio[:, 0]  # 提取指定的单通道
                if audio.size == 0:
                    print(f"Warning: {audio_file} is empty.")
                else:
                    concatenated_audio.append(audio)
            except Exception as e:
                print(f"Error reading {audio_file}: {e}")

        if not concatenated_audio:
            raise ValueError("No audio data to concatenate.")

        audio = self.feats_func(concatenated_audio[0])
        concatenated_audio = torch.from_numpy(np.column_stack(concatenated_audio)).float().reshape(8,-1) # [8,80000]
        x_stft = self.stft(concatenated_audio)
        # print(f'x_stft:{x_stft.shape}')
        # print(f'x_stft:{x_stft.shape}') # [64,8,257,497] 
        ipd_set=[]
        mic_pair_indexes = [(0,4),(1,5),(2,6),(3,7)]
        mic_pair_indexes = [tuple(map(int,pair)) for pair in mic_pair_indexes]
        csipd = True
        for i,j in mic_pair_indexes:
            # print(i,j)
            ipd = x_stft[:,i,:].angle() - x_stft[:,j,:].angle()

            # print(f'ipd:{ipd.shape}')
            if csipd:
                ipd = torch.cat([torch.sin(ipd),torch.cos(ipd)],dim=0)
            ipd_set.append(ipd)
        array_fea = torch.cat(ipd_set,dim=0)
    
        audio_Pxy = torch.cat((audio, array_fea), dim=0)
    
        label = label[:audio.shape[-1]]
        return audio_Pxy ,torch.from_numpy(label).long() ,torch.ones(len(label)).bool()


    

       



if __name__ == "__main__":

    import yaml
    with open("test.yml", "r") as f:
        confs = yaml.load(f,Loader=yaml.FullLoader)


    # a = OnlineChunkedFeats(chime6_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/audio/",split="train", 
    # label_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/", configs=confs)

    # a = OnlineFeats(ami_audio_root= "/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/audio/", 
    # label_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/train/" , configs=confs,probs=[0., 1.0])
    # a = OnlineFeats(ami_audio_root= "/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/audio/", 
    # label_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/eval/" , configs=confs)

    a = OnlineFeats_test(ami_audio_root= "/mnt/sdb1/lsj/datasets/AMI/media/sam/bx500/amicorpus/audio_eval/", 
    label_root="/mnt/sdb1/lsj/datasets/AMI/media/sam/bx500/amicorpus/fa_labels/eval/" , configs=confs)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(a, batch_size=1, shuffle=False)
    epoch = 1
    segment = 500
    for i in range(epoch):
        for  _, data in enumerate(dataloader):
            audio, label,mask = data
            
            print(f'audio.shape:{audio.shape}')
            print(f'label.shape:{label.shape}')
            # print(f"audio:{audio}")
            # print(f"label:{label}")

            # img, label = img.to(device), label.to(device).squeeze()
            # opt.zero_grad()
            # logits = model(img)
            # loss = criterion(logits, label)
            # pass
    # for i in DataLoader(a, batch_size=3, shuffle=True):
    #     print(f'ssdasd')
        # print(i)




