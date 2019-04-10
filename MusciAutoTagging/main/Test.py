import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

import time
import sys
import os
sys.path.append('../net')
sys.path.append('../pre')
from Mmnet import *
from data_gentor import *
import glob
from te_ext import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

Xtr, Ytr, Xva, Yva, Xte, Yte, avg_std = load_data('')

#a = mel(fn[0])

class HDF52Torch(Dataset):
    def __init__(self, X, Y, mode='Test'):
        self.X = X
        self.Y = Y
        self.mode = mode
    def __getitem__(self, index):

        rX = self.X[index]
        rY = self.Y[index]

        mX = torch.from_numpy(rX.astype('float32'))
        mY = torch.from_numpy(rY)
        return mX, mY
    
    def __len__(self):
        return len(self.X)

def show_model_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]
    print 'Model:' + model.module.model_name + '\t#params:%d'%(params)


class Trainer:
    def __init__(self, args):
        self.fn = glob.glob('../web_AED/static/fwave/*')
        model = nn.DataParallel(Net(args.mel, Ytr.shape[1]).cuda())
        self.model = model
        self.args = args

        # load avg and std for Z-score 
        Xavg = torch.tensor([avg_std[0]])
        Xstd = torch.tensor([avg_std[1]])
        self.Xavg, self.Xstd = Variable(Xavg.cuda()), Variable(Xstd.cuda())

        self.load_pretrained_model()
        
    def load_pretrained_model(self):
        # pre-training
        if os.path.exists(self.args.pmp):
            pretrained_model = torch.load(self.args.pmp)
            self.pretrained_model = pretrained_model
            model_param = self.model.state_dict()
            for k in pretrained_model['state_dict'].keys():
               try:
                    model_param[k].copy_(pretrained_model['state_dict'][k])
               except:
                    print '[ERROR] Load pre-trained model'
                    self.model.apply(model_init)
                    break
            print 'Load Pre_trained Model : ' + self.args.pmp
        
        else:
            print 'Learning from scrath'
            #self.model.apply(model_init)
            

    def predictor(self):
        st = time.time()
        all_pred = []
        self.model.eval()
        for i in self.fn:
            X = mel(i)
            X = torch.from_numpy(X.reshape(1, X.shape[0], X.shape[1]))
            print X.size()
            X = Variable( X.cuda())
            clip_out, _ = self.model(X, self.Xavg, self.Xstd)
            all_pred.extend(F.sigmoid(clip_out).data.cpu().numpy())

        print 'Prediction Time:%1f'%(time.time() - st)
        return np.array(all_pred)
    

