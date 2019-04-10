import time
import numpy as np
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
import multiprocessing
import functools

def class_F1_R_P(gru, pre, th):
    best = np.zeros(4)
    for t in th:
        tidx = gru==1
        vpred = pre.copy()
        vpred[vpred> t] = 1
        vpred[vpred<=t] = 0
        
        TP = vpred[tidx].sum()
        if TP == 0 :
            continue
        
        P = TP / float(vpred.sum())
        R = TP / float(gru.sum())
        F1 = 2*(P*R)/(R+P)
        
        if F1 > best[1]:
            best = np.array([t, F1, R, P])
    return best

def multi_evl_nt(i, gru, pre, th):
    st = time.time()
    evl_metrics = np.zeros(6)
    
    if gru[:,i].sum() == 0 or gru[:,i].sum()==len(gru):
        evl_metrics = evl_metrics -1
        return evl_metrics
    
    pre_tag =  (np.argmax(pre[:,:,i],1)==i).astype(int)
    evl_metrics[:4] = class_F1_R_P(gru[:,i], pre_tag, [0])
    
    evl_metrics[4] = average_precision_score(gru[:,i], pre[:,i,i])
    evl_metrics[5] = roc_auc_score(gru[:,i], pre[:,i,i])
    #print time.time() - st
    return evl_metrics

def multi_evl(i, gru, pre, th):
    st = time.time()
    evl_metrics = np.zeros(6)
    
    if gru[:,i].sum() == 0 or gru[:,i].sum()==len(gru):
        evl_metrics = evl_metrics -1
        return evl_metrics
    
    if len(th) == 0:
        #th = np.arange(0, 1, 0.0001)
        th = np.arange(0, 1, 0.01)
        evl_metrics[:4] = class_F1_R_P(gru[:,i], pre[:,i], th)
    else:
        #if len(th) == 1:
        #    evl_metrics[:4] = class_F1_R_P(gru[:,i], pre[:,i], th)
        #else:
        evl_metrics[:4] = class_F1_R_P(gru[:,i], pre[:,i], [th[i]])
    
    evl_metrics[4] = average_precision_score(gru[:,i], pre[:,i])
    evl_metrics[5] = roc_auc_score(gru[:,i], pre[:,i])
    #print time.time() - st
    return evl_metrics

def evl(gru, pre, va_th=[]):
    st =time.time()
    vate = 'TE'
    evl_metrics = np.zeros((pre.shape[-1], 6))
    if len(va_th) == 0:
        vate = 'VA'
    
    if vate not in ['TE', 'VA']:
        multi_evl_1 = functools.partial(multi_evl, gru=gru, pre=pre, th=va_th)
        P = multiprocessing.Pool(30)
        evl_metrics = np.array(P.map(multi_evl_1, np.arange(pre.shape[-1])))
        P.close()
        P.join()

    else:
        for i in np.arange(pre.shape[-1]):
            if len(pre.shape)==2:
                evl_metrics[i] = multi_evl(i, gru=gru, pre=pre, th=va_th)
            else:
                evl_metrics[i] = multi_evl_nt(i, gru=gru, pre=pre, th=va_th)
    
    va_th = evl_metrics[:,0].copy()
    evl_metrics = evl_metrics[:,1:] 
    
    #print np.arange(527)[evl_metrics[:,0]!=-1]
    acc = evl_metrics[evl_metrics[:,0]!=-1,:].mean(axis=0) * 100
    #print acc
    #print np.arange(pre.shape[-1])[evl_metrics[:,0]==-100,:]
    out = '[%s] mAP:%.1f%% AUC:%.1f%% F1-CB:%.1f%% R-CB:%.1f%% P-CB:%.1f%% time:%.1f'\
            % (vate, acc[3], acc[4], acc[0], acc[1], acc[2], time.time()-st)
    print out
    return va_th, evl_metrics, out


