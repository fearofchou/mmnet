import numpy as np
import os
import subprocess as subp
import multiprocessing
import ext
import functools

def run_mp(args, li):
    lv, sec, sr, ws, mel, evl = args
    i = li[:-1].replace('"','').replace(' ','').split(',')
    wav_fn = wav_fp[evl] + 'Y'+i[0]+'_'+i[1]+'_'+i[2]+'.wav'

    return ext.mel(wav_fn, lv=lv, sec=sec, sr=sr, ws=ws, mel=mel)

def get_tag(li):
    i = li[:-1].replace('"','').replace(' ','').split(',')

    # get Y tag
    Y = np.zeros(len(cl_dict))
    for tag in i[3:]:
        Y[cl_dict[tag]] = 1

    return Y

music_mood = np.arange(276,283)
music_genre = np.arange(216,265)
music = np.append(music_mood, music_genre)

# load data
as_csv = {}
with open('/home/fearofchou/ND/m189/max/FCNN_torch/pre/csv/balanced_train_segments.csv', 'r') as f:
    as_csv['bala_valid'] = f.readlines()
with open('/home/fearofchou/ND/m189/max/FCNN_torch/pre/csv/unbalanced_train_segments.csv', 'r') as f:
    as_csv['unbala_train'] = f.readlines()
with open('/home/fearofchou/ND/m189/max/FCNN_torch/pre/csv/eval_segments.csv', 'r') as f:
    as_csv['eval_test'] = f.readlines()
with open('/home/fearofchou/ND/m189/max/FCNN_torch/pre/csv/class_labels_indices.csv', 'r') as f:
    cl = f.readlines()

# get tag dict
cl_dict = {}
for i in cl[1:]:
    i= i.split(',')
    iid = i[0]
    mask_id = i[1]
    cl_dict[mask_id] = int(iid)

# find wav fn
wav_fp = {}
wav_fp['bala_valid'] = '/home/fearofchou/ND/data2/audioset_sy/audio/balance/'
wav_fp['eval_test'] = '/home/fearofchou/ND/data2/audioset_sy/audio/Test/'
wav_fp['unbala_train'] = '/home/fearofchou/ND/data2/audioset_sy/audio/seg/'

for i in wav_fp.keys():
    print i
    file_list = np.unique(as_csv[i])[3:]
    args = [10000, 10, 44100, 2048, 128, i]
    f_run_mp = functools.partial(run_mp, args)

    P = multiprocessing.Pool(20)
    Y = np.array(P.map(get_tag, file_list[:]))
    P.close()
    P.join()
    
    idx = np.arange(len(Y))[Y[:,music].sum(1)!=0]
    print 'number of files %d'%(len(idx))
    X = np.zeros((len(file_list[idx]), 128, 862))

    for i in xrange( (X.shape[0]/1000) +1):
        print 'load batch index %d %d'%(i, X.shape[0]/1000)
        P = multiprocessing.Pool(20)
        tmp_X = np.array(P.map(f_run_mp, file_list[idx[i*1000:(i+1)*1000]]))
        P.close()
        P.join()
        X[i*1000:(i+1)*1000] = tmp_X

    len_X = X.sum(1).sum(1)
    X_idx = np.arange(len(file_list[idx]))[len_X!=0]

    print 'Save file at ../data/AS_music_%s'%(args[-1])
    c = Y[idx][X_idx][:,music]

    np.save('../data/ASmusic_%s_Y'%(args[-1]), Y[idx][X_idx][:,music])
    np.save('../data/ASmusic_%s_X'%(args[-1]), X[X_idx])



