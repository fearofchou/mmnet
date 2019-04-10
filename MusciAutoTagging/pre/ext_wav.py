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

    return y



