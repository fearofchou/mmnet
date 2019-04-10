import numpy as np
import librosa
import os

def mel(fn, sr=44100, sec=10, lv=10000, ws=2048, mel=128):

    fea_len = int((sr*sec/(ws/4)))+1

    y, sr = librosa.load(fn, sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=ws,
                                       hop_length=ws/4, n_mels=mel)
    return np.log(1 + lv * S).astype('float32')




