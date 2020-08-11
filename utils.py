import torch
from torch import nn
import numpy as np

import librosa
import math
import scipy
from scipy.signal import lfilter
#import matplotlib.pyplot as plt
from spectrum import poly2lsf, lsf2poly
import pyworld as pw


EPSILON = 1e-2

def linear_quantize(samples, q_levels):
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return samples.long()

def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1

def q_zero(q_levels):
    return q_levels // 2

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)

def vocoder(prev_samples, M):
    [N, C, L] = prev_samples.size()
    hf = torch.zeros([N, C, M]) 

    _temp_seq_pitch = torch.reshape(prev_samples,(N, C*L)).cpu().double().numpy()

    for batch_ind in range(N):
        temp_seq_pitch = _temp_seq_pitch[batch_ind,:]

        # extraction du pitch
        fs = 16000
        _f0, t = pw.dio(temp_seq_pitch, fs)    # raw pitch extractor
        f0_ = pw.stonemask(temp_seq_pitch, _f0, t, fs)  # pitch refinement
        f0 = f0_[1::2]
        ind = t[1::2] * fs

        pitch = torch.from_numpy(np.asarray(f0))

        for frame_ind in range(C):
            temp_seq = prev_samples[batch_ind, frame_ind, :].cpu().numpy()

            # Calculs lsf
            # Calcul du bruit blanc gaussien à ajouter au signal pour éviter "ill-conditioned" matrix
            # rms = np.sqrt(np.mean(temp_seq**2))
            # print(rms)
            # n_next = 1
            # if rms == 0:
            #     while rms == 0:
            #         temp_seq = prev_samples[batch_ind, frame_ind+n_next, :].cpu().numpy()
            #         rms = np.sqrt(np.mean(temp_seq**2))
            #         n_next += 1
            #         print("flag")

            rms = 0.1
            var = rms * 0.0001 # (-40db)
            std = math.sqrt(var)
            mu, sigma = 0, std # mean = 0 and standard deviation

            temp_seq = temp_seq + np.random.normal(mu, sigma, L)
        
            # hanning windowing
            temp_seq = temp_seq*np.hanning(L)

            a = librosa.core.lpc(temp_seq, M-3)  # lpc of order M-3 
            lsf = torch.from_numpy(np.asarray(poly2lsf(a)))

            # Calculs nrj
            residu = lfilter(a, 1, temp_seq)
            nrjRMS_residu = torch.from_numpy(np.asarray(np.sqrt(np.mean(residu**2))))

            # Calculs Voising flag
            voice_flag = torch.tensor([0])
            if pitch[frame_ind] >= 50 and pitch[frame_ind] <= 300: # condition for 
                voice_flag = torch.tensor([1])

            hf[batch_ind, frame_ind, :-3] = lsf
            hf[batch_ind, frame_ind, -3] = nrjRMS_residu
            hf[batch_ind, frame_ind, -2] = pitch[frame_ind]
            hf[batch_ind, frame_ind, -1] = voice_flag
            
    return hf.cuda()

