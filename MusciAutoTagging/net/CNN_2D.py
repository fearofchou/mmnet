import torch.nn as nn
import torch.nn.functional as F
import torch
class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()
        self.conv = nn.Conv2d(inp, out, (3,3), padding=(1,1), bias=False)
        self.bn = nn.BatchNorm2d(out)       

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out


class Net(nn.Module):
    def __init__(self, channel, num_classes):
        super(Net, self).__init__()
        self.nc = num_classes
        self.model_name = 'CNN_2D'

        self.ks = (3,3)
        self.ps = (1,1)
        
        inp = 64*2
        self.b1 = block(    1,    16)
        self.b2 = block(   16, inp*2)
        self.b3 = block(inp*2, inp*4)
        
        self.det1 = nn.Conv2d(   16, self.nc, self.ks, padding=self.ps, bias=False)
        self.det2 = nn.Conv2d(inp*2, self.nc, self.ks, padding=self.ps, bias=False)
        self.det3 = nn.Conv2d(inp*4, self.nc, self.ks, padding=self.ps, bias=False)
        self.att1 = nn.Conv2d(   16,    16, self.ks, padding=self.ps, bias=False)
        self.att2 = nn.Conv2d(inp*2, inp*2, self.ks, padding=self.ps, bias=False)
        self.att3 = nn.Conv2d(inp*4, inp*4, self.ks, padding=self.ps, bias=False)
        
        self.dp = nn.Dropout(.5)
        dns = 512*2
        
        # linear
        #self.den1 = nn.Linear(inp*3*2, dns)
        #self.den2 = nn.Linear(dns, dns)
        #self.dbn1 = nn.BatchNorm1d(dns)       
        #self.dbn2 = nn.BatchNorm1d(dns)       
        #self.prd = nn.Linear(dns, self.nc * 2)
        self.prd = nn.Linear(inp*4*2, self.nc * 1)
    
    def nn_apl(self, x):
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
    def nn_att(self, inp, att):
        att_out = F.softmax(att(inp).view(inp.size(0), inp.size(1), -1), dim=-1)
        
        att_sc = att_out.sum(1).view(inp.size(0), 1, inp.size(2), inp.size(3))
        att_sc = att_sc.repeat(1, self.nc, 1, 1)
        #att_ens = F.softmax(att_sup, dim=3)
        
        return att_sc, (att_out*inp.view(inp.size(0), inp.size(1), -1)).sum(-1)

    def forward(self, x, Xavg, Xstd):
    #def forward(self, x):
        xs = x.size()
        x = x.view(xs[0], 1, xs[1], xs[2])
        
        #x = x.permute(0,2,1,3)
        #Xavg = Xavg.view(1, Xavg.size(0),1,1).repeat(xs[0], 1, xs[2], xs[3])
        #Xstd = Xstd.view(1, Xstd.size(0),1,1).repeat(xs[0], 1, xs[2], xs[3])
        z_x = (x - Xavg)/Xstd
        
        
        b1  = self.b1(z_x)
        mp1 = F.max_pool2d(b1, (4,4))
        b2 = self.b2(mp1)
        mp2 = F.max_pool2d(b2, (4,4))
        bf  = self.b3(mp2)
        
        # global average pooling
        gap = self.nn_apl(bf)
        att1, _ = self.nn_att(b1, self.att1)
        att2, _ = self.nn_att(b2, self.att2)
        att3, att_embed = self.nn_att(bf, self.att3)
        gap = torch.cat([gap, att_embed], dim=1)

        #DNN
        #den1 = F.relu(self.dbn1(self.den1(self.dp(gap))))
        #den2 = F.relu(self.dbn2(self.den2(self.dp(den1))))
        y_cls = self.prd(self.dp(gap))
            
        # attention
        det1 = self.det1(self.dp(b1))
        det2 = self.det2(self.dp(b2))
        det3 = self.det3(self.dp(bf))
        

        # ensemble prediction
        att_ens = F.softmax(att3.view(att3.size(0), att3.size(1), -1), dim=-1)
        y_att = (det3.view(det3.size(0), det3.size(1), -1) * att_ens).sum(-1)
        y_ens = (y_cls + y_att)/2
        

        return y_ens, [[att1, det1], [att2, det2], [att3, det3]]
        #return y_ens, []
        #return y_cls, [[att, det]]



