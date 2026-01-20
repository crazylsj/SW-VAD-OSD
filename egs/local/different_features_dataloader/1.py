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


class OnlineFeats(Dataset):

    def __init__(self, ami_audio_root, label_root, configs, segment=500, probs=None, synth=None):

        self.configs = configs
        self.segment = segment
        self.probs = probs
        self.synth = None
        self.mic_pair_indexes = [(0,4),(1,5),(2,6),(3,7)]

        audio_files = glob.glob(os.path.join(ami_audio_root, "**/*.wav"), recursive=True)
        # print(f'audio_files:{audio_files}')
        for f in audio_files:
            if len(sf.SoundFile(f)) < self.segment:
                print("Dropping file {}".format(f))
        labels = glob.glob(os.path.join(label_root, "*.wav"))
        # print(labels)
        lab_hash = {}

        for l in labels:
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

        #assert len(set(list(meta.keys())).difference(set(list(lab_hash.keys())))) == 0
        # remove keys

        self.label_hash = lab_hash #-保存所有的标签路径，格式：{‘IS1008b’:''/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/dev/LABEL-IS1008b.wav'}
        # print(f'----------------------------------self.label_hash:{self.label_hash }--------------------------------')

        # print(f'self.probs:{self.probs}') #probs:[0.3,0.7]
        if self.probs: # parse for data-augmentation
            label_one = []
            label_two = []

            for l in labels:
                c_label, _ = sf.read(l)  # read it all
          
                # l:/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/train/LABEL-ES2013a.wav
                # c_label:[0. 0. 0. ... 0. 0. 0.],len:82518
                sess = Path(l).stem.split("-")[-1]
                # find contiguous
                tmp = self.get_segs(c_label, 1, 1)
                for s,e in tmp:
                    assert not np.where(c_label[s:e] > 1)[0].any()
                tmp = [(sess, x[0], x[1]) for x in tmp]  # we need session also
                label_one.extend(tmp)

                # do the same for two speakers
                tmp = self.get_segs(c_label, 2, 2)
                for s, e in tmp:
                    assert not np.where(c_label[s:e] != 2)[0].any()
                tmp = [(sess, x[0], x[1]) for x in tmp]
                label_two.extend(tmp)

            self.label_one = label_one
            self.label_two = label_two
            # print(f'self.label_one:{self.label_one}')  # ('EN2001a', 426508, 426627) 将一个人说话的片段和两个人说话的片段an'z
            # print(f'self.label_two:{self.label_two}')

        self.tot_length = int(np.sum([len(sf.SoundFile(l)) for l in labels]) / segment)

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
        return self.tot_length



    def segment_signal(self, signal, frame_size, overlap):
        step_size = frame_size - overlap
        num_frames = (len(signal) - overlap) // step_size
        if num_frames <= 0:
            return np.empty((0, frame_size))
        frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::step_size]
        return frames

    def apply_window(self, frames, window):
        if frames.size == 0:
            return frames
        return frames * window

    def compute_symmetric_csd(self, array_segments, fs, nperseg, overlap):
        if nperseg <= 0 or overlap < 0 or overlap >= nperseg:
            raise ValueError("Invalid values for nperseg or overlap")

        num_segments = len(array_segments)
        Pxy_results = []

        for i in range(num_segments):
            for j in range(i, num_segments):  # 计算对称矩阵的上三角部分
                # print(f'ij:{i}{j}')
                
                f, Pxy = csd(array_segments[i], array_segments[j], fs=fs, window='hann', nperseg=nperseg, noverlap=overlap)
                Pxy_results.append(Pxy)
                # print(f'Pxy:{Pxy.shape}')
                # print(f'len(Pxy_results):{len(Pxy_results)}')


        return Pxy_results

  
    def stft(self, x, fs, frame_length, hop_length, window):
        # 计算每帧的样本数
        frame_length_samples = int(frame_length * fs / 1000)
        hop_length_samples = int(hop_length * fs / 1000)

        # 计算总的帧数
        num_frames = (len(x) - frame_length_samples) // hop_length_samples + 1

        # 初始化STFT矩阵
        stft_matrix = np.empty((frame_length_samples // 2 + 1, num_frames), dtype=np.complex64)

        # 加窗并计算每帧的FFT
        for i in range(num_frames):
            start = i * hop_length_samples
            # print(f'start:{start}')
            frame = x[start:start + frame_length_samples]
            windowed_frame = frame * window
            stft_matrix[:, i] = np.fft.rfft(windowed_frame)

        return stft_matrix
        
    def extract_top_stft(self, stft_matrix, top_indices):
        num_top_freqs, num_frames = top_indices.shape
         

        # 使用高级索引一次性提取所有需要的 STFT 值
        # np.arange(num_frames) 生成一个列索引的数组
        # top_indices.flatten() 生成所有需要的频率索引的数组
        extracted_stft = stft_matrix[top_indices.flatten(), np.tile(np.arange(num_frames), num_top_freqs)]

        # 重塑为 (num_top_freqs, num_frames) 形状
        extracted_stft = extracted_stft.reshape(num_top_freqs, num_frames)
        return extracted_stft

        # # 初始化矩阵，用于存储提取出的 STFT 值
        # extracted_stft = np.zeros((num_top_freqs, num_frames), dtype=stft_matrix.dtype)
        
        # # 对每一帧进行提取
        # for frame in range(num_frames):
        #     indices = top_indices[:, frame]
        #     # print(f'indices:{indices}')
        #     # 从 stft_matrix 中提取出对应频率的傅里叶变换结果
        #     extracted_stft[:, frame] = stft_matrix[indices, frame]
        #     # print(f'extracted_stft:{extracted_stft}')

        # return extracted_stft

    def compute_cross_spectrum(self,X, Y):
        """
        计算两个频谱特征的互功率谱。
        
        参数：
        X : array_like
            第一个信号的频谱特征 (STFT, FFT 等)。
        Y : array_like
            第二个信号的频谱特征 (STFT, FFT 等)。
        
        返回值：
        Pxy : array_like
            两个信号的互功率谱。
        """
        # 确保 Y 是 X 的共轭复数
        Pxy = X * np.conjugate(Y)
        
        return Pxy

    def re_compute_symmetric_csd(self, array_segments):


        num_segments = len(array_segments)
        Pxy_results = []

        for i in range(num_segments):
            for j in range(i, num_segments):  # 计算对称矩阵的上三角部分
                # print(f'ij:{i}{j}')
                # print(f'array_segments[i]:{array_segments[i].shape}')
                
                Pxy = self.compute_cross_spectrum(array_segments[i], array_segments[j])
                # print(f'Pxy:{Pxy.shape}')
                Pxy_results.append(Pxy)
                # print(f'Pxy:{Pxy.shape}')
                # print(f'len(Pxy_results):{len(Pxy_results)}')


        return Pxy_results
        
    def opp_compute_symmetric_csd(self, array_segments):
        ipd_set=[]
        for i,j in self.mic_pair_indexes:
            # print(i,j)
            Pxy = self.compute_cross_spectrum(array_segments[i], array_segments[j])
        
            ipd_set.append(Pxy)
        return ipd_set




    def noaugm(self):
        try:
            
            file = np.random.choice(self.devices)
            sess = Path(file).stem.split(".")[0]
            
            label_path = self.label_hash[sess]
            with sf.SoundFile(label_path) as f:
                file_length = len(f)
            
            start = np.random.randint(0, file_length - self.segment - 2)
            stop = start + self.segment
            label, _ = sf.read(label_path, start=start, stop=stop)
            
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
            
            start = int(start * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
            stop = int(stop * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])

            # ----------------------------------所有方法都要用的---------------------
            audio_list = self.devices_hash[sess]
            # print(f'sess:{sess}')
            # print(f'audio_list:{audio_list}')
            array1_audio_list = sorted([audio_path for audio_path in audio_list if 'Array1' in audio_path])
            concatenated_audio = []
            for audio_file in array1_audio_list:
                # print(f'audio_file:{audio_file}')
                try:
                    audio, fs = sf.read(audio_file, start=start, stop=stop)
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

            concatenated_audio = np.column_stack(concatenated_audio).T #(8,80000) len(concatenated_audio):8
            # print(f'concatenated_audio:{concatenated_audio.shape}')


            # -----------------------------------------------------------------
            
            fs = 16000  # 采样率 (Hz)
            frame_length = 25  # 帧长 (ms)
            hop_length = 10  # 帧移 (ms)
            window = np.hanning(int(frame_length * fs / 1000))  # Hann窗
            re_stft = []
            
            max_k = 5

            stft_matrix = self.stft(concatenated_audio[0], fs, frame_length, hop_length, window)
            magnitude = np.abs(stft_matrix) #(201, 498)
            # print(f'magnitude:{magnitude[:,0]}')

            # print(f'np.argsort(magnitude, axis=0):{np.argsort(magnitude, axis=0)[:,0]}')
            top_indices = np.argpartition(magnitude, -max_k, axis=0)[-max_k:]  # 获取前10个最大值的索引
            sorted_indices = np.argsort(-magnitude[top_indices, np.arange(magnitude.shape[1])], axis=0)

            # 重新排列 top_indicesf，使其对应的 magnitude 值是从大到小排列
            top_indices = top_indices[sorted_indices, np.arange(magnitude.shape[1])]
    
            for i in range(len(concatenated_audio)):
                # 计算STFT
                stft_matrix = self.stft(concatenated_audio[i], fs, frame_length, hop_length, window)
                re_stft_matrix = self.extract_top_stft(stft_matrix,top_indices)
                # print(f're_stft_matrix:{re_stft_matrix.shape}')
                re_stft.append(re_stft_matrix)

            Pxy_results = self.opp_compute_symmetric_csd(re_stft)
            _ , frames = Pxy_results[0].shape

            # print(f'Pxy_results:{Pxy_results[0].shape}')

            combined_array = np.stack(Pxy_results, axis=-1).reshape(-1,frames)

            concatenated_Pxy_real = np.real(combined_array).astype("float32")
            concatenated_Pxy_imag = np.imag(combined_array).astype("float32")
            
            if not Pxy_results:
                raise ValueError("No valid Pxy results.")
            
   

            concatenated_Pxy_real = torch.from_numpy(concatenated_Pxy_real)
            concatenated_Pxy_imag = torch.from_numpy(concatenated_Pxy_imag)
            pxy_ri = torch.cat((concatenated_Pxy_real, concatenated_Pxy_imag), dim=0)
           

            audio = self.feats_func(concatenated_audio[0])
            audio_Pxy = torch.cat((audio, pxy_ri), dim=0)

        
            label = label[:audio.shape[-1]]

            return audio_Pxy, torch.from_numpy(label).long(), torch.ones(len(label)).bool()
        
        except Exception as e:
            print("An error occurred:", e)
            traceback.print_exc()



    @staticmethod
    def normalize(signal, target_dB):

        fx = (AudioEffectsChain().custom(
            "norm {}".format(target_dB)))
        signal = fx(signal)
        return signal

    def __getitem__(self, item):
        return self.noaugm()

       

class OnlineChunkedFeats(Dataset):

    def __init__(self, chime6_root, split, label_root, configs, segment=300):

        self.configs = configs
        self.segment = segment
        meta = parse_chime6(chime6_root, split)

        devices = {}
        for sess in meta.keys():
            devices[sess] = []
            for array in meta[sess]["arrays"].keys():
                devices[sess].extend(meta[sess]["arrays"][array]) # only channel 1

        labels = glob.glob(os.path.join(label_root, "*.wav"))
        lab_hash = {}

        for l in labels:
            l_sess = str(Path(l).stem).split("-")[-1]
            lab_hash[l_sess] = l

        self.lab_hash = lab_hash
        chunks = self.get_chunks(labels)

        examples = []
        for sess in chunks.keys():
            for s, e in chunks[sess]:
                for dev in devices[sess]:
                    examples.append((dev, s, e))

        self.examples = examples

        self.set_feats_func()

    def set_feats_func(self):

        # initialize feats_function
        if self.configs["feats"]["type"] == "mfcc_kaldi":
            from torchaudio.compliance.kaldi import mfcc
            self.feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                             **self.configs["mfcc_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "fbank_kaldi":
            from torchaudio.compliance.kaldi import fbank
            self.feats_func = lambda x: fbank(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                              **self.configs["fbank_kaldi"]).transpose(0, 1)
        elif self.configs["feats"]["type"] == "spectrogram_kaldi":
            from torchaudio.compliance.kaldi import spectrogram
            self.feats_func = lambda x: spectrogram(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                                    **self.configs["spectrogram_kaldi"]).transpose(0, 1)
        else:
            raise NotImplementedError

    def get_chunks(self, labels):

        chunks = {}
        chunk_size = self.configs["data"]["segment"]
        frame_shift = self.configs["data"]["segment"]

        for l in labels:
            sess = Path(l).stem.split("-")[-1]
            chunks[sess] = []
            # generate chunks for this file
            c_length = len(sf.SoundFile(l)) # get the length of the session files in samples
            for st, ed in _gen_frame_indices(
                    c_length, chunk_size, frame_shift, use_last_samples=False):
                chunks[sess].append([st, ed])
        return chunks


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        device, s, e = self.examples[item]
     
        sess = get_session(device)
        labelfile = self.lab_hash[sess]

        label, _ = sf.read(labelfile, start=s, stop=e)
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

        start = int(s * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
        stop = int(e * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"] +
                   self.configs["data"]["fs"] * self.configs["feats"]["hop_size"] * 2)

        audio, fs = sf.read(device, start=start, stop=stop)

        if len(audio.shape) > 1:  # binaural
            audio = audio[:, np.random.randint(0, 1)]

        audio = self.feats_func(audio)
        assert audio.shape[-1] == len(label)
        return audio, torch.from_numpy(label).long()


if __name__ == "__main__":

    import yaml
    with open("train.yml", "r") as f:
        confs = yaml.load(f,Loader=yaml.FullLoader)


    # a = OnlineChunkedFeats(chime6_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/audio/",split="train", 
    # label_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/", configs=confs)

    # a = OnlineFeats(ami_audio_root= "/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/audio/", 
    # label_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/train/" , configs=confs,probs=[0., 1.0])
    # a = OnlineFeats(ami_audio_root= "/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/audio/", 
    # label_root="/sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/media/sam/bx500/amicorpus/fa_labels/eval/" , configs=confs)

    a = OnlineFeats(ami_audio_root= "/datasets/AMI/media/sam/bx500/amicorpus/audio/", 
    label_root="/datasets/AMI/media/sam/bx500/amicorpus/fa_labels/eval" , configs=confs)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(a, batch_size=1, shuffle=True)
    epoch = 1
    for i in range(epoch):
        for  audio, label in enumerate(dataloader):
            print(f'1')
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




