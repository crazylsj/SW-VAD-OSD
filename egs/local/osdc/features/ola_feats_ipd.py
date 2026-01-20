import torch
import numpy as np
from osdc.utils.oladd import _gen_frame_indices
from scipy.signal import csd, windows
import soundfile as sf
import torch.nn.functional as F
def compute_feats_windowed(feats_func, audio, winsize=16000*30, stride=(16000*30-160*2)):

    res = []
    # print(f'len(audio):{len(audio)}')
    # print(f'_gen_frame_indices(len(audio), winsize, stride , True):{_gen_frame_indices(len(audio), winsize, stride , True)}')
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        # print(f'st, ed :{st},{ed}')
        tmp = feats_func(audio[st:ed])
        # print(f'tmp:{tmp.shape}')
        #tmp = tmp if st == 0 else tmp[:, 160*10:]
        res.append(tmp)

    if type(res[0]) == torch.Tensor:
        out = torch.cat(res, -1)
    else:
        out = np.concatenate(res, -1)

    return out

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
        # return torch.stack(X).permute(1, 0, 2, 3)  # (batch, channel, freq, time)

def compute_feats_windowed_ipd(feats_func, audio, wav, winsize=16000*30, stride=(16000*30-160*2)):
    stft = MultichannelSTFT()

    res = []
    # concatenated_audio = []
    single_audio = audio   
   

    # print(f'len(audio):{len(audio)}')
    # print(f'_gen_frame_indices(len(audio), winsize, stride , True):{_gen_frame_indices(len(audio), winsize, stride , True)}')
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        # print(f'st, ed :{st},{ed}')
        # print(f'audio:{audio.shape}')
        # single_tmp = feats_func(single_audio[st:ed])
        concatenated_audio = []
        
        # print(f'tmp:{tmp.shape}')
        #tmp = tmp if st == 0 else tmp[:, 160*10:]
        for audio_file in wav:
            audio, fs = sf.read(audio_file, start=st, stop=ed)
            if audio.ndim > 1:
                audio = audio[:, 0]  # 提取指定的单通道
            if audio.size == 0:
                print(f"Warning: {audio_file} is empty.")
            else:
                concatenated_audio.append(audio)

        # concatenated_audio = np.column_stack(concatenated_audio)

        audio = feats_func(concatenated_audio[0])
        audio_array = torch.from_numpy(np.column_stack(concatenated_audio)).float().reshape(8,-1) # [8,80000]

        x_stft = stft(audio_array)
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
    
        tmp = torch.cat((audio, array_fea), dim=0)
        # print(f'pxy_ri:{pxy_ri.shape}')
        # print(f'tmp:{tmp.shape}')

        res.append(tmp)

    if type(res[0]) == torch.Tensor:
        out = torch.cat(res, -1)
    else:
        out = np.concatenate(res, -1)

    return out
    
if __name__ == "__main__":

    from gammatone import get_gammatonegm_

    random = np.random.random((16000*30))

    gammas = lambda x : get_gammatonegm_(x)
    windowed = compute_feats_windowed(gammas, random, 16000*4, 16000*4-160*2)

    direct = gammas(random)

    np.testing.assert_array_almost_equal(direct, windowed)