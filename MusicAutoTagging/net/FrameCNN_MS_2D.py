import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../fun')
from layers import *

class Net(nn.Module):
    def __init__(self, channel, ks, num_labels):
        super(Net, self).__init__()
        self.numl = num_labels
        self.channel = channel
        ks = 128
        self.model_name = 'FrameCNN_MS_GAP_DNN'
        
        fss = 16
        self.conv1 = nn.Conv2d(1, fss, (3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(fss)       
        self.conv2 = nn.Conv2d(fss, ks*2, (3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(ks*2)       
        self.conv3 = nn.Conv2d(ks*2, ks*4, (3,3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(ks*4)       
        
        self.dconv1 = nn.Conv2d(ks*4, ks*2, (3,3), padding=(1,1))
        self.db1 = nn.BatchNorm2d(ks*2)       
        self.dconv2 = nn.Conv2d(ks*2, fss, (3,3), padding=(1,1))
        self.db2 = nn.BatchNorm2d(fss)       
        
        fs = (3,3)
        ps = (1,1)
        self.det1 = nn.Conv2d(fss, self.numl, fs, padding=ps)
        self.det2 = nn.Conv2d(ks*2, self.numl, fs, padding=ps)
        self.det3 = nn.Conv2d(ks*4, self.numl, fs, padding=ps)
        self.det4 = nn.Conv2d(ks*2, self.numl, fs, padding=ps)
        self.det5 = nn.Conv2d(fss, self.numl, fs, padding=ps)
        
        self.sou = nn.Conv2d(fss, 1, fs, padding=ps)
        
        self.dp = nn.Dropout(.0)
        dns = 512*1
        
        # linear
        self.channel = ks*4*1
        self.den1 = nn.Linear(self.channel, dns)
        self.den2 = nn.Linear(dns, dns)
        self.dbn1 = nn.BatchNorm1d(dns)       
        self.dbn2 = nn.BatchNorm1d(dns)       
        self.channel = dns
        #self.channel = ks*4

        self.prd = nn.Linear(self.channel, self.numl)
    
    def apl(self, x):
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
    def GAP(self, layer, att, pool=0):
        # (50, 384, 56, 1)
        # (50, 1, 56, 1)
        out = att(layer).permute(0,2,1,3)
        # (50, 56, 1, 1)
        out = F.softmax(out)
        # (50, 56, 1, 1)
        out = out.permute(0,2,1,3)
        # (50, 1, 56, 1)
        #otu = att.repeat(1, bf.size(1), 1, 1)
        # (50, 384, 56, 1)
        if pool == 1:
            out1 = out * layer
            # (50, 384)
            out1 = out1.sum(-1).sum(-1)
        else:
            out1 = 1
        
        out = out.sum(1).view(out.size(0), 1, layer.size(2), 1)

        out2 = out.permute(0,2,1,3)
        out2 = F.softmax(out2)
        out2 = out2.permute(0,2,1,3)
        out2 = out2.repeat(1, self.numl, 1, 1)
        
        out = out.repeat(1, self.numl, 1, 1)

        return out1, out, out2

    def forward(self, x, Xavg, Xstd):
        
        #x = x.permute(0,3,2,1)

        xs = x.size()
        Xavg = Xavg.view(1, Xavg.size()[0],1,1).repeat(xs[0], 1, xs[2], xs[3])
        Xstd = Xstd.view(1, Xstd.size()[0],1,1).repeat(xs[0], 1, xs[2], xs[3])
        z_x = (x - Xavg)/Xstd
        
        ms = (8,16)
        c1 = F.relu(self.bn1(self.conv1(z_x)))
        m1 = F.max_pool2d(c1, ms)
        c2 = F.relu(self.bn2(self.conv2(m1)))
        m2 = F.max_pool2d(c2, ms)
        c3 = F.relu(self.bn3(self.conv3(m2)))
        
        '''
        uc3 = F.upsample(c3, scale_factor=ms, mode='nearest')
        c4 = F.relu(self.db1(self.dconv1(uc3)))
        uc4 = F.upsample(c4, scale_factor=ms, mode='nearest')
        c5 = F.relu(self.db2(self.dconv2(uc4)))
        '''
        d1 = self.det1(c1)
        d2 = self.det2(c2)
        d3 = self.det3(c3)
        #d4 = self.det4(c4)
        #d5 = self.det5(c5)
        
        #s = F.sigmoid(self.sou(c5))
        #s = F.relu(self.sou(c5))

        # pooling
        apl = self.apl(c3)
        #apl = l_tpl(c3)
        den1 = F.relu(self.dbn1(self.den1(self.dp(apl))))
        den2 = F.relu(self.dbn2(self.den2(self.dp(den1))))
        #den2 = F.relu(self.dbn2(self.den2(self.dp(den1))))
        pred = self.prd(den2)
        #pred = self.prd(apl)
        
        #return pred, [pred, d1, d2, d3, d4, d5, x], [pred, d1,d2,d3,d4,d5,s*x],[]
        #return pred, [pred, d1, d2, d3, d4, d5], [pred, d1,d2,d3,d4,d5],[]
        return pred, [pred], [pred],[]



