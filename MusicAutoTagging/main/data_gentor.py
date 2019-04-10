import numpy as np
import SharedArray as sa
import sys
import h5py

def f_load(m_name, fp):
    try:
        out = sa.attach(m_name)
    except:
        print 'load data : %s'%(fp)
        out = np.load(fp)
        X = sa.create(m_name, (out.shape), dtype='float32')
        X[:] = out
    return out.astype('float32')

def load_data(data_name):
    va_Y = f_load('ASmusic_va_Y' , '../data/ASmusic_bala_valid_Y.npy')
    va_X = f_load('ASmusic_va_X' , '../data/ASmusic_bala_valid_X.npy')
    te_Y = f_load('ASmusic_te_Y' , '../data/ASmusic_eval_test_Y.npy')
    te_X = f_load('ASmusic_te_X' , '../data/ASmusic_eval_test_X.npy')
    tr_Y = f_load('ASmusic_tr_Y' , '../data/ASmusic_unbala_train_Y.npy')
    tr_X = f_load('ASmusic_tr_Y' , '../data/ASmusic_unbala_train_X.npy')

    avg_std = np.load('../data/AS_ubl_X_avg_std.npy')

    print 'load data : ASmusic_music'
    return tr_X, tr_Y, va_X, va_Y, te_X, te_Y, avg_std
        
        

