import torch
import numpy as np
from osdc.utils.oladd import _gen_frame_indices
from scipy.signal import csd, windows
import soundfile as sf

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

def segment_signal(signal, frame_size, overlap):
    step_size = frame_size - overlap
    num_frames = (len(signal) - overlap) // step_size
    if num_frames <= 0:
        return np.empty((0, frame_size))
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::step_size]
    return frames


def compute_feats_windowed_array(feats_func, audio, wav, winsize=16000*30, stride=(16000*30-160*2)):

    res = []
    # concatenated_audio = []
    single_audio = audio   
   

    # print(f'len(audio):{len(audio)}')
    # print(f'_gen_frame_indices(len(audio), winsize, stride , True):{_gen_frame_indices(len(audio), winsize, stride , True)}')
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        # print(f'st, ed :{st},{ed}')
        # print(f'audio:{audio.shape}')
        single_tmp = feats_func(single_audio[st:ed])
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

        concatenated_audio = np.column_stack(concatenated_audio)
            
        fs = 16000
        nperseg = 400
        noverlap = 240

        frame_size = nperseg
        overlap = nperseg - noverlap
        window = windows.hann(frame_size)
        array_segments = [segment_signal(concatenated_audio[:, i], frame_size, noverlap) for i in range(concatenated_audio.shape[1])]

        Pxy_results = []
        for i in range(0, 8):
            if array_segments[i].size == 0:
                # print(f"Skipping empty segment {i}")
                continue
            f, Pxy = csd(array_segments[0], array_segments[i], fs=fs, window='hann', nperseg=nperseg, noverlap=overlap, nfft=nperseg)
            # print(f'Pxy:{Pxy}')
            Pxy_results.append(Pxy)
        
        if not Pxy_results:
            raise ValueError("No valid Pxy results.")
        
        min_shape = min(Pxy.shape for Pxy in Pxy_results)
        Pxy_results = [Pxy[:min_shape[0], :min_shape[1]] for Pxy in Pxy_results]

        concatenated_Pxy_real = np.concatenate([Pxy.real for Pxy in Pxy_results], axis=1).astype("float32")
        concatenated_Pxy_imag = np.concatenate([Pxy.imag for Pxy in Pxy_results], axis=1).astype("float32")
  

        concatenated_Pxy_real = torch.from_numpy(concatenated_Pxy_real).transpose(0, 1)
        concatenated_Pxy_imag = torch.from_numpy(concatenated_Pxy_imag).transpose(0, 1)
        pxy_ri = torch.cat((concatenated_Pxy_real, concatenated_Pxy_imag), dim=0)
        tmp = torch.cat((single_tmp, pxy_ri), dim=0)
        # print(f'pxy_ri:{pxy_ri.shape}')
        # print(f'tmp:{tmp.shape}')

        res.append(tmp)

    if type(res[0]) == torch.Tensor:
        out = torch.cat(res, -1)
    else:
        out = np.concatenate(res, -1)

    return out

def stft(x, fs, frame_length, hop_length, window):
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
        
def extract_top_stft(stft_matrix, top_indices):
    num_top_freqs, num_frames = top_indices.shape

    # 初始化矩阵，用于存储提取出的 STFT 值
    extracted_stft = np.zeros((num_top_freqs, num_frames), dtype=stft_matrix.dtype)
    
    # 对每一帧进行提取
    for frame in range(num_frames):
        indices = top_indices[:, frame]
        # print(f'indices:{indices}')
        # 从 stft_matrix 中提取出对应频率的傅里叶变换结果
        extracted_stft[:, frame] = stft_matrix[indices, frame]
        # print(f'extracted_stft:{extracted_stft}')

    return extracted_stft

def compute_cross_spectrum(X, Y):
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




def re_compute_symmetric_csd(array_segments):
    num_segments = len(array_segments)
    Pxy_results = []

    for i in range(num_segments):

        for j in range(i, num_segments):  # 计算对称矩阵的上三角部分
            # print(f'ij:{i}{j}')
            # print(f'array_segments[i]:{array_segments[i].shape}')
            
            Pxy = compute_cross_spectrum(array_segments[i], array_segments[j])
            # print(f'Pxy:{Pxy.shape}')
            Pxy_results.append(Pxy)
            # print(f'Pxy:{Pxy.shape}')
            # print(f'len(Pxy_results):{len(Pxy_results)}')

    return Pxy_results

def compute_feats_windowed_array_csd8_topmax10(feats_func, audio, wav, winsize=16000*30, stride=(16000*30-160*2)):

    res = []
    # concatenated_audio = []
    single_audio = audio   
   

    # print(f'len(audio):{len(audio)}')
    # print(f'_gen_frame_indices(len(audio), winsize, stride , True):{_gen_frame_indices(len(audio), winsize, stride , True)}')
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        # print(f'st, ed :{st},{ed}')
        # print(f'audio:{audio.shape}')
        single_tmp = feats_func(single_audio[st:ed])
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

        concatenated_audio = np.column_stack(concatenated_audio).T

        fs = 16000  # 采样率 (Hz)
        frame_length = 25  # 帧长 (ms)
        hop_length = 10  # 帧移 (ms)
        window = np.hanning(int(frame_length * fs / 1000))  # Hann窗
        re_stft = []
        
        max_k = 5

        stft_matrix = stft(concatenated_audio[0], fs, frame_length, hop_length, window)
        magnitude = np.abs(stft_matrix) #(201, 498)
        # print(f'magnitude:{magnitude[:,0]}')

        # print(f'np.argsort(magnitude, axis=0):{np.argsort(magnitude, axis=0)[:,0]}')
            
            
        top_indices = np.argsort(magnitude, axis=0)[-max_k:]  # 获取前10个最大值的索引
        top_indices = np.flip(top_indices, axis=0)  # 翻转，使得最大值在前面
        # print(f'top_indices:{top_indices[:,0]}')
        for i in range(len(concatenated_audio)):
            # 计算STFT
            stft_matrix = stft(concatenated_audio[i], fs, frame_length, hop_length, window)
            re_stft_matrix = extract_top_stft(stft_matrix,top_indices)
            # print(f're_stft_matrix:{re_stft_matrix.shape}')
            re_stft.append(re_stft_matrix)
        Pxy_results = re_compute_symmetric_csd(re_stft)
        _ , frames = Pxy_results[0].shape
        combined_array = np.stack(Pxy_results, axis=-1).reshape(-1,frames)

        concatenated_Pxy_real = np.real(combined_array).astype("float32")
        concatenated_Pxy_imag = np.imag(combined_array).astype("float32")
        
        if not Pxy_results:
            raise ValueError("No valid Pxy results.")
        


        concatenated_Pxy_real = torch.from_numpy(concatenated_Pxy_real)
        concatenated_Pxy_imag = torch.from_numpy(concatenated_Pxy_imag)
        pxy_ri = torch.cat((concatenated_Pxy_real, concatenated_Pxy_imag), dim=0)
            
        tmp = torch.cat((single_tmp, pxy_ri), dim=0)
        # print(f'pxy_ri:{pxy_ri.shape}')
        # print(f'tmp:{tmp.shape}')

        res.append(tmp)

    if type(res[0]) == torch.Tensor:
        out = torch.cat(res, -1)
    else:
        out = np.concatenate(res, -1)

    return out

def opp_compute_symmetric_csd(array_segments):

    ipd_set=[]
    mic_pair_indexes = [(0,4),(1,5),(2,6),(3,7)]
    for i,j in mic_pair_indexes:
        # print(i,j)
        Pxy = compute_cross_spectrum(array_segments[i], array_segments[j])
    
        ipd_set.append(Pxy)
    return ipd_set

def compute_feats_windowed_frequency_selected_diagonal_csd(feats_func, audio, wav, winsize=16000*30, stride=(16000*30-160*2)):

    res = []
    # concatenated_audio = []
    single_audio = audio   
   

    # print(f'len(audio):{len(audio)}')
    # print(f'_gen_frame_indices(len(audio), winsize, stride , True):{_gen_frame_indices(len(audio), winsize, stride , True)}')
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        # print(f'st, ed :{st},{ed}')
        # print(f'audio:{audio.shape}')
        
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

        concatenated_audio = np.column_stack(concatenated_audio).T

        fs = 16000  # 采样率 (Hz)
        frame_length = 25  # 帧长 (ms)
        hop_length = 10  # 帧移 (ms)
        window = np.hanning(int(frame_length * fs / 1000))  # Hann窗
        re_stft = []
        
        max_k = 5

        stft_matrix = stft(concatenated_audio[0], fs, frame_length, hop_length, window)
        magnitude = np.abs(stft_matrix) #(201, 498)
        # print(f'magnitude:{magnitude[:,0]}')

        # print(f'np.argsort(magnitude, axis=0):{np.argsort(magnitude, axis=0)[:,0]}')
            
            
        top_indices = np.argsort(magnitude, axis=0)[-max_k:]  # 获取前10个最大值的索引
        top_indices = np.flip(top_indices, axis=0)  # 翻转，使得最大值在前面
        # print(f'top_indices:{top_indices[:,0]}')
        for i in range(len(concatenated_audio)):
            # 计算STFT
            stft_matrix = stft(concatenated_audio[i], fs, frame_length, hop_length, window)
            re_stft_matrix = extract_top_stft(stft_matrix,top_indices)
            # print(f're_stft_matrix:{re_stft_matrix.shape}')
            re_stft.append(re_stft_matrix)
        Pxy_results = opp_compute_symmetric_csd(re_stft)
        _ , frames = Pxy_results[0].shape
        combined_array = np.stack(Pxy_results, axis=-1).reshape(-1,frames)

        # concatenated_Pxy_real = np.real(combined_array).astype("float32")
        # concatenated_Pxy_imag = np.imag(combined_array).astype("float32")
        
        # if not Pxy_results:
        #     raise ValueError("No valid Pxy results.")
        


        concatenated_Pxy_real = torch.from_numpy(np.real(combined_array).astype("float32"))
        concatenated_Pxy_imag = torch.from_numpy(np.imag(combined_array).astype("float32"))
        pxy_ri = torch.cat((concatenated_Pxy_real, concatenated_Pxy_imag), dim=0)

        single_tmp = feats_func(concatenated_audio[0])
            
        tmp = torch.cat((single_tmp, pxy_ri), dim=0)
        # print(f'pxy_ri:{pxy_ri.shape}')
        # print(f'tmp:{tmp.shape}')

        res.append(tmp)

    if type(res[0]) == torch.Tensor:
        out = torch.cat(res, -1)
    else:
        out = np.concatenate(res, -1)

    return out

# def opp_compute_symmetric_csd(array_segments):
#     ipd_set=[]
#     for i,j in mic_pair_indexes:
#         # print(i,j)
#         Pxy = self.compute_cross_spectrum(array_segments[i], array_segments[j])
    
#         ipd_set.append(Pxy)
#     return ipd_set

def compute_feats_windowed_diagonal_csd(feats_func, audio, wav, winsize=16000*30, stride=(16000*30-160*2)):

    res = []
    # concatenated_audio = []
    single_audio = audio   
   

    # print(f'len(audio):{len(audio)}')
    # print(f'_gen_frame_indices(len(audio), winsize, stride , True):{_gen_frame_indices(len(audio), winsize, stride , True)}')
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        # print(f'st, ed :{st},{ed}')
        # print(f'audio:{audio.shape}')
        
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

        concatenated_audio = np.column_stack(concatenated_audio).T
        fs = 16000  # 采样率 (Hz)
        frame_length = 25  # 帧长 (ms)
        hop_length = 10  # 帧移 (ms)
        window = np.hanning(int(frame_length * fs / 1000))  # Hann窗

        re_stft = []

        for i in range(len(concatenated_audio)):
            # 计算STFT
            stft_matrix = stft(concatenated_audio[i], fs, frame_length, hop_length, window)
            re_stft.append(stft_matrix)
    

        Pxy_results = opp_compute_symmetric_csd(re_stft)
        _ , frames = Pxy_results[0].shape

        # print(f'Pxy_results:{Pxy_results[0].shape}')

        combined_array = np.stack(Pxy_results, axis=-1).reshape(-1,frames)
    
        concatenated_Pxy_real = torch.from_numpy(np.real(combined_array).astype("float32"))
        concatenated_Pxy_imag = torch.from_numpy(np.imag(combined_array).astype("float32"))
        pxy_ri = torch.cat((concatenated_Pxy_real, concatenated_Pxy_imag), dim=0)

        single_tmp = feats_func(concatenated_audio[0])
            
        tmp = torch.cat((single_tmp, pxy_ri), dim=0)
        # print(f'pxy_ri:{pxy_ri.shape}')
        # print(f'tmp:{tmp.shape}')

        res.append(tmp)

    if type(res[0]) == torch.Tensor:
        out = torch.cat(res, -1)
    else:
        out = np.concatenate(res, -1)

    return out

def compute_symmetric_csd(array_segments, fs, nperseg, overlap):
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

def compute_feats_windowed_csd8_array(feats_func, audio, wav, winsize=16000*30, stride=(16000*30-160*2)):

    res = []
    # concatenated_audio = []
    single_audio = audio   
   

    # print(f'len(audio):{len(audio)}')
    # print(f'_gen_frame_indices(len(audio), winsize, stride , True):{_gen_frame_indices(len(audio), winsize, stride , True)}')
    for st, ed in _gen_frame_indices(len(audio), winsize, stride , True):
        # print(f'st, ed :{st},{ed}')
        # print(f'audio:{audio.shape}')
        single_tmp = feats_func(single_audio[st:ed])
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

        concatenated_audio = np.column_stack(concatenated_audio)
            
        fs = 16000
        nperseg = 400
        noverlap = 240

        frame_size = nperseg
        overlap = nperseg - noverlap
        window = windows.hann(frame_size)
        array_segments = [segment_signal(concatenated_audio[:, i], frame_size, noverlap) for i in range(concatenated_audio.shape[1])]

        Pxy_results = compute_symmetric_csd(array_segments=array_segments,fs=fs,nperseg=nperseg,overlap=overlap)
        # print(f'Pxy_results:{Pxy_results.shape}')
        combined_array = np.mean(np.stack(Pxy_results, axis=-1), axis=1)
        # print(f'combined_array:{combined_array.shape}')
        concatenated_Pxy_real = np.real(combined_array).astype("float32")
        concatenated_Pxy_imag = np.imag(combined_array).astype("float32")
        
        if not Pxy_results:
            raise ValueError("No valid Pxy results.")

        concatenated_Pxy_real = torch.from_numpy(concatenated_Pxy_real).transpose(0, 1)
        concatenated_Pxy_imag = torch.from_numpy(concatenated_Pxy_imag).transpose(0, 1)
        pxy_ri = torch.cat((concatenated_Pxy_real, concatenated_Pxy_imag), dim=0)

   
        tmp = torch.cat((single_tmp, pxy_ri), dim=0)
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