import numpy as np
import librosa
import subprocess as subp

def get_wav_ffmpeg(fn, sr):
    command = ['ffmpeg', '-i', fn, '-f', 'f32le',
            '-acodec', 'pcm_f32le', '-ar', str(sr),
            '-loglevel', 'quiet','-ac', str(1), '-']
    pipe = subp.Popen(command, stdout=subp.PIPE,
                startupinfo=None)
    raw_audio = pipe.stdout.read()
    return np.fromstring(raw_audio, dtype="float32")

def get_wav_librosa(fn, sr):
    y, _ = librosa.core.load(fn, sr)
    return y

def get_padded_fea(fea, max_time_len):
    fea_time_len = fea.shape[1]
    
    if fea_time_len == max_time_len:
        return fea

    if fea_time_len > max_time_len:
        return fea[:,:max_time_len]

    if fea_time_len < max_time_len:
        # only pad time axis
        return np.pad(fea, ((0,0),(0,max_time_len - fea_time_len)), 'constant')

def get_mel_spect(fn, args):
    max_time_len = int((args.sr*args.msc)/args.hs)
    
    wav = get_wav_librosa(fn, args.sr)

    if len(wav) == 0:
        print 'Read Empty Wave File %s'%(fn)
        return -2 # Empty wave file 
    else:
        fea = librosa.feature.melspectrogram(wav, sr=args.sr, n_fft=args.ws, 
                hop_length=args.hs, n_mels=args.mel)
        pad_fea = get_padded_fea(fea, max_time_len)

    return np.log(1 + 10000 * pad_fea).reshape(1, args.mel, max_time_len)

