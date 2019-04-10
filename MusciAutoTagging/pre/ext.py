import numpy as np
import librosa
import os

def mel(fn, sr=44100, sec=10, lv=10000, ws=2048, mel=128):

    fea_len = int((sr*sec/(ws/4)))+1
    init_mel = np.zeros((mel, fea_len))

    if not os.path.isfile(fn):
        #print fn
        return init_mel

    y, sr = librosa.load(fn, sr)

    if len(y) < 4410 :
        print fn
        print len(y)
        return init_mel


    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=ws,
                                       hop_length=ws/4, n_mels=mel)
    ss = S.shape[1]
    if fea_len == ss:
        init_mel = S.copy()
    if fea_len < ss:
        init_mel = S[:,:fea_len].copy()
    if fea_len > ss:
        init_mel[:, :ss] = S.copy()

    init_mel = np.log(1 + lv * init_mel)

    return init_mel



