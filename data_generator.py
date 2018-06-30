from feature_extractor import *
import multiprocessing as mu
from functools import partial
import h5py
import time 
import os
import sys

def get_tr_set_avg_std(data):
    st = time.time()
    print 'Get std and average of training set'
    
    common_sum = 0
    square_sum = 0
    
    # data.shape -> (1, 128, 896)
    number_non_zero_len = 0
    for i in xrange(len(data)):
        # ignore zero padding
        number_non_zero_len += (data[i].sum((0,1))!=0).astype(int).sum()
        common_sum += data[i].sum(0).sum(-1)
        square_sum += (data[i]**2).sum(0).sum(-1)
        
    common_avg = common_sum / number_non_zero_len
    square_avg = square_sum / number_non_zero_len
    
    std = np.sqrt( square_avg - common_avg**2 )
    print 'length of std' + str(std.shape)
    print time.time() - st
    return np.array([common_avg, std])

def get_csv_file(fn):
    with open(fn) as f:
        data = f.readlines()
    return data

def get_tag2idx(fn):
    #idx2tag = {}
    tag2idx = {}
    classes = get_csv_file(fn)
    for idx, val in enumerate(classes[1:]):
        mask_tag_id, confidence, tag_name = val[:-1].split('\t')
        tag2idx[tag_name] = idx

    return tag2idx

def get_fn2target(fn, tag2idx, wav_path):
    # target.shape -> (number of files, 17) binary vectors
    fn2target = {}
    for i in get_csv_file(fn):
        wav_fn, start_time, end_time, tag_name = i[:-1].split('\t')
        wav_fn = wav_path + '/Y' + wav_fn
        try:
            fn2target[wav_fn][tag2idx[tag_name]] = 1
        except:
            fn2target[wav_fn] = np.zeros(len(tag2idx))
            fn2target[wav_fn][tag2idx[tag_name]] = 1
    return fn2target

def get_create_h5_data(h5, set_name,data_len, num_classes, args):
    max_time_len = int((args.sr*args.msc)/args.hs)
    Xtr = h5.create_dataset('X'+set_name, 
            shape=(data_len, 1, args.mel, max_time_len),
            maxshape=(None, 1, args.mel, max_time_len),
            chunks=(1, 1, args.mel, max_time_len), dtype='float32')
    Ytr = h5.create_dataset('Y'+set_name, (data_len, num_classes), dtype='int')
    return Xtr, Ytr

def get_data_fea_target_to_h5(h5, data2target, set_name, num_classes, args):
    X, Y = get_create_h5_data(h5, set_name, len(data2target), num_classes, args)
    
    # put target (Y) to h5
    file_list = np.array(data2target.keys())
    Y[:] = np.array(data2target.values()).copy()

    # put mel spect (X) to h5
    process_files = 0
    multi_read_files = 488
    st = time.time()
    get_mel_spect_partial = partial(get_mel_spect, args=args)
    for i in xrange(np.ceil(len(data2target)/float(multi_read_files)).astype(int)):
        #P = mu.Pool(mu.cpu_count())
        P = mu.Pool(30)
        out = np.array(P.map(get_mel_spect_partial, file_list[i*multi_read_files:(i+1)*multi_read_files]))
        P.close()
        P.join()
        X[i*multi_read_files:(i+1)*multi_read_files] = out
        process_files += len(out)
        sys.stdout.write('\r')
        sys.stdout.write('Extract feature for each %s example [%7d/%7d] Time %d'
                %(set_name, process_files, len(file_list), time.time()-st))
        sys.stdout.flush()
        
    #print '\n'
    print '\nExtract feature for %s set: Done'%(set_name)
    return X, Y


def get_h5_dataset(h5_fn, tr_csv_fn, te_csv_fn, tr_wav_fp, te_wav_fp, classes_fn, args):
    
    # Create h5 file
    if os.path.isfile(h5_fn):
        print '[File Exist] Read file : %s'%(h5_fn)
        return h5py.File(h5_fn, 'r')
   
   # get tag index
    tag2idx = get_tag2idx(classes_fn)
    target_len = len(tag2idx)
    
    tr2target = get_fn2target(tr_csv_fn, tag2idx, tr_wav_fp)
    te2target = get_fn2target(te_csv_fn, tag2idx, te_wav_fp)

    h5f = h5py.File(h5_fn, 'w')
    Xte, Yte = get_data_fea_target_to_h5(h5f, te2target, 'te', len(tag2idx), args)
    Xtr, Ytr = get_data_fea_target_to_h5(h5f, tr2target, 'tr', len(tag2idx), args)
    
    # get training set std&avg
    avg_std = h5f.create_dataset('Xtr_avg_std',
            data=get_tr_set_avg_std(Xtr), dtype='float32')
    h5f.close()

    print 'H5 File Path %s'%(h5_fn)
    return h5py.File(h5_fn, 'r')


