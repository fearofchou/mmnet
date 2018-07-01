import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

import time
import sys
import os
from evaluator import *

class HDF52Torch(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __getitem__(self, index):
        mX = torch.from_numpy(self.X[index])
        mY = torch.from_numpy(self.Y[index])
        return mX, mY
    
    def __len__(self):
        return len(self.X)

def model_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)

def show_model_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]
    print 'Model:' + model.module.model_name + '\t#params:%d'%(params)


class Trainer:
    def __init__(self, data, model, args):
        self.model = model
        self.args = args

        # data builder
        data_args = {'batch_size': args.bs, 'num_workers': 0, 'pin_memory': True}
        self.tr_loader = torch.utils.data.DataLoader(HDF52Torch(data['Xtr'], data['Ytr']), 
                shuffle=True, drop_last=True, **data_args)
        self.va_loader = torch.utils.data.DataLoader(HDF52Torch(data['Xte'], data['Yte']), 
                  **data_args)

        # if you want to make a evaluation on evaluation set, please build a data_loader for it
        # self.te_loader = torch.utils.data.DataLoader(HDF52Torch(data['Xte'][:], data['Yte'][:]), 
                  #**data_args)
        
        # load avg and std for Z-score 
        Xavg = torch.from_numpy(data['Xtr_avg_std'][0].astype('float32'))
        Xstd = torch.from_numpy(data['Xtr_avg_std'][1].astype('float32'))
        self.Xavg, self.Xstd = Variable(Xavg.cuda()), Variable(Xstd.cuda())

        # pre-class loss weight
        # http://www.cs.tut.fi/sgn/arg/dcase2017/documents/workshop_presentations/the_story_of_audioset.pdf
        class_prior = data['Ytr'][:].sum(0) / float(data['Ytr'][:].sum())
        mean_prior = class_prior.mean()
        PCLW = ( (mean_prior/ class_prior) * ((1-mean_prior)/(1-class_prior)) )**args.beta
        self.PCLW = torch.from_numpy(PCLW.astype('float32')).cuda()
        print self.PCLW   
        self.show_dataset_model_params()
        self.load_pretrained_model()
        
    def load_pretrained_model(self):
        # pre-training
        if os.path.exists(self.args.pmp):
            pretrained_model = torch.load(self.args.pmp)
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
            self.model.apply(model_init)
            

    def show_dataset_model_params(self):
        # show model structure
        print self.model
        
        # show params
        print show_model_params(self.model)
        
        # show the size of training, validation and test set
        print 'Dataset : ' + self.args.dn
        print 'Xtr->' + str(self.tr_loader.dataset.X.shape) + '\t\tYtr->' + str(self.tr_loader.dataset.Y.shape)
        print 'Xte->' + str(self.va_loader.dataset.X.shape) + '\t\tYva->' + str(self.va_loader.dataset.Y.shape)
        #print 'Xte->' + str(self.te_loader.dataset.X.shape) + '\t\tYte->' + str(self.te_loader.dataset.Y.shape)

    def mm_loss(self, target, macro_out, micro_out):
        
        #tar = target.data
        target = target.float()
        we = self.PCLW
        wwe = self.args.gw
        we *= wwe
        loss = 0
        we = we.view(1,-1).repeat(target.size(0), 1)
        
        twe = we * target + (1 - target)*wwe

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=twe, size_average=True)
        loss += loss_fn(macro_out, target)
        
        for att_sc, det in micro_out:
            os = det.size()
            fl_target = target.view(os[0], os[1], 1, 1).repeat(1,1,os[2],os[3])
            twe = we.view(os[0],os[1],1,1).repeat(1, 1, os[2], os[3])
            twe = att_sc.data * twe * fl_target + (1 - fl_target) * wwe
            # Noet: att_sc.data is requirement
            
            loss_fn = torch.nn.BCEWithLogitsLoss(weight=twe, size_average=True)
            loss += loss_fn(det, fl_target)
        
        return loss


    def predictor(self, loader):
        st = time.time()
        all_pred = []
        self.model.eval()
        for data, target in loader:
            with torch.no_grad():
                data, target = Variable(data.cuda()), Variable(target.cuda())
            clip_out, _ = self.model(data, self.Xavg, self.Xstd)
            all_pred.extend(F.sigmoid(clip_out).data.cpu().numpy())

        print 'Prediction Time:%1f'%(time.time() - st)
        return np.array(all_pred)
    
    def fit(self):
        st = time.time()
        save_dict = {}
        for e in xrange(1, self.args.ep+1):
            
            # set optimizer (SGD)
            lr = self.args.lr ** ((e/(self.args.lrde))+1) 
            print '\n==> Training Epoch #%d lr=%4f'%(e, lr)
            self.optimizer = optim.SGD(self.model.parameters(),
                    lr=lr, momentum=self.args.mom, weight_decay=self.args.wd)

            # Training
            for batch_idx, (data, target) in enumerate(self.tr_loader):
                data, target = Variable(data.cuda()), Variable(target.cuda())
                self.model.train()

                macro_out, micro_out = self.model(data, self.Xavg, self.Xstd)
                loss = self.mm_loss(target, macro_out, micro_out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # print training epoch, training loss and training time 
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d'
                        %(e, self.args.ep, batch_idx+1, len(self.tr_loader),
                            loss.item(), time.time() - st))
                sys.stdout.flush()
            print '\n'
            
            # evaluation
            all_pred = self.predictor(self.va_loader)
            va_class_threshold, _, va_out = evl(self.va_loader.dataset.Y, all_pred)
            
            save_dict['state_dict'] = self.model.state_dict()
            save_dict['tr_loss'] = loss
            save_dict['va_out'] = va_out
            save_dict['va_class_threshold'] = va_class_threshold
            

            # test on evaluation set and save the results
            ##########################
            #all_pred = self.predictor(self.te_loader)
            #_, te_result_pre_class, te_out = evl(self.te_loader.dataset.Y, all_pred, va_th=va_class_threshold)
            #save_dict['te_out'] = te_out
            #save_dict['te_result_pre_class'] = te_result_pre_class
            ##########################
            
            
            directory = './data/model/%s'%(self.model.module.model_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save(save_dict, directory +'/epoch_%d'%(e))


