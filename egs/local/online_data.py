from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from osdc.utils.oladd import _gen_frame_indices
from pysndfx import AudioEffectsChain
from scipy.signal import csd   
import soundfile as sf
import traceback


class OnlineFeats(Dataset):

    def __init__(self, ami_audio_root, label_root, configs, segment=500, synth=None):

        self.configs = configs
        self.segment = segment
        self.synth = None
        self.mic_pair_indexes = [(0,4),(1,5),(2,6),(3,7)]
        audio_files = glob.glob(os.path.join(ami_audio_root, "**/*.wav"), recursive=True)
        for f in audio_files:
            if len(sf.SoundFile(f)) < self.segment:
                print("Dropping file {}".format(f))
        labels = glob.glob(os.path.join(label_root, "*.wav"))
        lab_hash = {}

        for l in labels:
            l_sess = str(Path(l).stem).split("-")[-1]
            lab_hash[l_sess] = l
        devices_hash = {}
        devices = []
        for f in audio_files:
            sess = Path(f).stem.split(".")[0] 
            if sess not in lab_hash.keys():
                continue
            devices.append(f)
            if sess not in devices_hash.keys():
                devices_hash[sess] = [f]
            else:
                devices_hash[sess].append(f)
        self.devices = devices  
        self.devices_hash = devices_hash 
        self.label_hash = lab_hash 
        self.tot_length = int(np.sum([len(sf.SoundFile(l)) for l in labels]) / segment)
        self.set_feats_func()
        if synth:
            self.synth=synth
    

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
            if (changePoints[i + 1] - changePoints[i]) > 30:  
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
            for j in range(i, num_segments):   
                f, Pxy = csd(array_segments[i], array_segments[j], fs=fs, window='hann', nperseg=nperseg, noverlap=overlap)
                Pxy_results.append(Pxy)
        return Pxy_results

  
    def stft(self, x, fs, frame_length, hop_length, window):
 
        frame_length_samples = int(frame_length * fs / 1000)
        hop_length_samples = int(hop_length * fs / 1000)
        num_frames = (len(x) - frame_length_samples) // hop_length_samples + 1
        stft_matrix = np.empty((frame_length_samples // 2 + 1, num_frames), dtype=np.complex64)

        for i in range(num_frames):
            start = i * hop_length_samples
            frame = x[start:start + frame_length_samples]
            windowed_frame = frame * window
            stft_matrix[:, i] = np.fft.rfft(windowed_frame)

        return stft_matrix
        
    def extract_top_stft(self, stft_matrix, top_indices):
        num_top_freqs, num_frames = top_indices.shape
        extracted_stft = stft_matrix[top_indices.flatten(), np.tile(np.arange(num_frames), num_top_freqs)]
        extracted_stft = extracted_stft.reshape(num_top_freqs, num_frames)
        return extracted_stft


    def compute_cross_spectrum(self,X, Y):
        Pxy = X * np.conjugate(Y)
        return Pxy

    def re_compute_symmetric_csd(self, array_segments):
        num_segments = len(array_segments)
        Pxy_results = []

        for i in range(num_segments):
            for j in range(i, num_segments): 
                Pxy = self.compute_cross_spectrum(array_segments[i], array_segments[j])
                Pxy_results.append(Pxy)
        return Pxy_results
        
    def opp_compute_symmetric_csd(self, array_segments):
        ipd_set=[]
        for i,j in self.mic_pair_indexes:
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

            audio_list = self.devices_hash[sess]
            array1_audio_list = sorted([audio_path for audio_path in audio_list if 'Array1' in audio_path])
            concatenated_audio = []
            for audio_file in array1_audio_list:
                try:
                    audio, fs = sf.read(audio_file, start=start, stop=stop)
                    if audio.ndim > 1:
                        audio = audio[:, 0]  
                    if audio.size == 0:
                        print(f"Warning: {audio_file} is empty.")
                    else:
                        concatenated_audio.append(audio)
                except Exception as e:
                    print(f"Error reading {audio_file}: {e}")

            if not concatenated_audio:
                raise ValueError("No audio data to concatenate.")

            concatenated_audio = np.column_stack(concatenated_audio).T 

            
            fs = 16000  
            frame_length = 25  
            hop_length = 10  
            window = np.hanning(int(frame_length * fs / 1000))  
            re_stft = []
            max_k = 5
            stft_matrix = self.stft(concatenated_audio[0], fs, frame_length, hop_length, window)
            magnitude = np.abs(stft_matrix) 
            top_indices = np.argpartition(magnitude, -max_k, axis=0)[-max_k:] 
            sorted_indices = np.argsort(-magnitude[top_indices, np.arange(magnitude.shape[1])], axis=0)
            top_indices = top_indices[sorted_indices, np.arange(magnitude.shape[1])]
                        
            for i in range(len(concatenated_audio)):
                stft_matrix = self.stft(concatenated_audio[i], fs, frame_length, hop_length, window)
                re_stft_matrix = self.extract_top_stft(stft_matrix,top_indices)
                re_stft.append(re_stft_matrix)

            Pxy_results = self.opp_compute_symmetric_csd(re_stft)
            _ , frames = Pxy_results[0].shape

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

       




