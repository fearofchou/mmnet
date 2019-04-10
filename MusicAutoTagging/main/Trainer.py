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
from data_gentor import *
#from Mmnet import *
from CNN_2D import *
from evaluator import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

Xtr, Ytr, Xva, Yva, Xte, Yte, avg_std = load_data('')

class HDF52Torch(Dataset):
    def __init__(self, X, Y, mode='Test'):
        self.X = X
        self.Y = Y
        self.mode = mode
    def __getitem__(self, index):

        if self.mode == 'Training':
            # roll
            rate = np.random.randint(100, self.X[0].shape[-1] - 100)
            rX = np.roll(self.X[index], rate, axis=-1)
            rX = rX[:,:400]
            rY = self.Y[index]
            '''
            #mixup 
            midx = np.random.randint(0, len(self.X))
            mrate = np.random.randint(100, self.X[0].shape[-1] - 100)
            rmX = np.roll(self.X[midx], mrate, axis=-1)
            rX = (rX + rmX)/1
            rY = self.Y[index] + self.Y[midx]
            rY[rY>1] = 1
            '''
        else:
            rX = self.X[index]
            rY = self.Y[index]

        mX = torch.from_numpy(rX.astype('float32'))
        mY = torch.from_numpy(rY)
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
    def __init__(self, args):
        model = nn.DataParallel(Net(args.mel, Ytr.shape[1]).cuda())
        self.model = model
        self.args = args

        # data builder
        data_args = {'batch_size': args.bs, 'num_workers': 5, 'pin_memory': True}
        self.tr_loader = torch.utils.data.DataLoader(HDF52Torch(Xtr, Ytr, mode='Training'), 
                shuffle=True, drop_last=True, **data_args)
        data_args = {'batch_size': 16, 'num_workers': 5, 'pin_memory': True}
        self.va_loader = torch.utils.data.DataLoader(HDF52Torch(Xva, Yva), 
                  **data_args)
        self.te_loader = torch.utils.data.DataLoader(HDF52Torch(Xte, Yte), 
                  **data_args)

        # load avg and std for Z-score 
        Xavg = torch.tensor([avg_std[0]])
        Xstd = torch.tensor([avg_std[1]])
        self.Xavg, self.Xstd = Variable(Xavg.cuda()), Variable(Xstd.cuda())

        # pre-class loss weight
        # http://www.cs.tut.fi/sgn/arg/dcase2017/documents/workshop_presentations/the_story_of_audioset.pdf
        class_prior = Ytr[:].sum(0) / float(Ytr[:].sum())
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
            #self.model.apply(model_init)
            

    def show_dataset_model_params(self):
        # show model structure
        print self.model
        
        # show params
        print show_model_params(self.model)
        
        # show the size of training, validation and test set
        print 'Dataset : ' + self.args.dn
        print 'Xtr->' + str(self.tr_loader.dataset.X.shape) + '\t\tYtr->' + str(self.tr_loader.dataset.Y.shape)
        print 'Xva->' + str(self.va_loader.dataset.X.shape) + '\t\tYva->' + str(self.va_loader.dataset.Y.shape)
        print 'Xte->' + str(self.te_loader.dataset.X.shape) + '\t\tYte->' + str(self.te_loader.dataset.Y.shape)

    def mm_loss(self, target, macro_out, micro_out):
        
        #tar = target.data
        target = target.float()
        #iwe = 3 - self.PCLW
        we = self.PCLW * 1
        wwe = self.args.gw
        wwe = 1
        #we *= wwe
        loss = 0
        if (macro_out.size(1)/2) == target.size(1):
            we = we.view(1,-1).repeat(target.size(0), 1)
            macro_out = F.log_softmax(macro_out)
            l = (macro_out * target * we).sum() / target.sum()
            loss = -l

        else:
        #if (macro_out.size(1)/1) == target.size(1):
            
            we = we.view(1,-1).repeat(target.size(0), 1)
            iwe = 2 - we
            twe = we * target + (1 - target)*iwe
            loss_fn = torch.nn.BCEWithLogitsLoss(weight=twe, size_average=True)
            loss += loss_fn(macro_out, target)
        for att_sc, det in micro_out:
            os = det.size()
            fl_target = target.view(os[0], os[1], 1, 1).repeat(1,1,os[2],os[3])
            #twe = we.view(os[0],os[1],1,1).repeat(1, 1, os[2], os[3])
            #tiwe = iwe.view(os[0],os[1],1,1).repeat(1, 1, os[2], os[3])
            #twe = att_sc.data * twe * fl_target + (1 - fl_target) * tiwe
            #twe =  twe * fl_target + (1 - fl_target) * wwe
            # Noet: att_sc.data is requirement
            #loss_fn = torch.nn.BCEWithLogitsLoss(weight=twe, size_average=True)
            itwe = att_sc.data * twe.view(os[0], os[1], 1, 1).repeat(1,1,os[2],os[3])
            loss_fn = torch.nn.BCELoss(weight=itwe, size_average=True)
            l = loss_fn(F.sigmoid(det), fl_target)
            loss += l
        
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
            #mt = clip_out.max(1)[0]
            #clip_out =  clip_out / (mt.view(-1, 1).repeat(1, Ytr.shape[1]))
            #all_pred.extend(clip_out.data.cpu().numpy())
            #all_pred.extend(F.sigmoid(clip_out).data.cpu().numpy())
            #all_pred.extend(F.softmax(clip_out).data.cpu().numpy())

        print 'Prediction Time:%1f'%(time.time() - st)
        return np.array(all_pred)
    
    def fit(self):
        st = time.time()
        save_dict = {}
        self.model.train()
        for e in xrange(1, self.args.ep+1):
            
            # set optimizer (SGD)
            lr = self.args.lr * ( 0.1 **( e/self.args.lrde ))
            #lr = self.args.lr ** ((e/(self.args.lrde))+1) 
            print '\n==> Training Epoch #%d lr=%4f'%(e, lr)
            self.optimizer = optim.SGD(self.model.parameters(),
                    lr=lr, momentum=self.args.mom, weight_decay=self.args.wd)

            # Training
            for batch_idx, (data, target) in enumerate(self.tr_loader):
                data, target = Variable(data.cuda()), Variable(target.cuda())
                
                macro_out, micro_out = self.model(data, self.Xavg, self.Xstd)
                #macro_out, micro_out = self.model(data)
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
            all_pred = self.predictor(self.te_loader)
            _, te_result_pre_class, te_out = evl(self.te_loader.dataset.Y, all_pred, va_th=va_class_threshold)
            print va_class_threshold
        
            save_dict['state_dict'] = self.model.state_dict()
            save_dict['tr_loss'] = loss
            save_dict['te_out'] = te_out
            save_dict['va_class_threshold'] = va_class_threshold

            # test on evaluation set and save the results
            ##########################
            #all_pred = self.predictor(self.te_loader)
            #_, te_result_pre_class, te_out = evl(self.te_loader.dataset.Y, all_pred, va_th=va_class_threshold)
            #save_dict['te_out'] = te_out
            #save_dict['te_result_pre_class'] = te_result_pre_class
            ##########################
            
            
            directory = '../model/%s'%(self.model.module.model_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save(save_dict, directory +'/epoch_%d'%(e))


