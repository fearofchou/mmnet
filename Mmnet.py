import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp)       
        self.conv1 = nn.Conv2d(inp, out, (1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(out)       
        self.conv2 = nn.Conv2d(out, out, (1,3), padding=(0,1))
        self.bn3 = nn.BatchNorm2d(out)       

        self.sk = nn.Conv2d(inp, out, (1,1), padding=(0,0))
    def forward(self, x):
        out = self.bn1(x)
        bn1 = F.relu(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.sk(x)
        return out, bn1


class Net(nn.Module):
    def __init__(self, channel, num_classes):
        super(Net, self).__init__()
        self.nc = num_classes
        self.model_name = 'Mmnet_1D'

        self.ks = (1,3)
        self.ps = (0,1)
        inp = channel
            
        self.conv1 = nn.Conv2d(inp, inp, self.ks, padding=self.ps)
        self.b1 = block(inp*1, inp*2)
        self.b2 = block(inp*2, inp*3)
        self.b3 = block(inp*3, inp*3)
        self.bnf = nn.BatchNorm2d(inp*3)       
        
        self.det = nn.Conv2d(inp*3, self.nc, self.ks, padding=self.ps)
        self.att = nn.Conv2d(inp*3, inp*3, self.ks, padding=self.ps)
        
        self.dp = nn.Dropout(.0)
        dns = 512*2
        
        # linear
        self.den1 = nn.Linear(inp*3, dns)
        self.den2 = nn.Linear(dns, dns)
        self.dbn1 = nn.BatchNorm1d(dns)       
        self.dbn2 = nn.BatchNorm1d(dns)       
        self.channel = dns

        self.prd = nn.Linear(self.channel, self.nc)
    
    def nn_apl(self, x):
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
    def nn_att(self, inp, att):
        att_out = F.softmax(att(inp), dim=3)
        
        att_sc = att_out.sum(1).view(inp.size(0), 1, inp.size(2), inp.size(3))
        att_sc = att_sc.repeat(1, self.nc, 1, 1)
        #att_ens = F.softmax(att_sup, dim=3)
        
        return att_sc

    def forward(self, x, Xavg, Xstd):
        
        x = x.permute(0,2,1,3)
        xs = x.size()
        Xavg = Xavg.view(1, Xavg.size(0),1,1).repeat(xs[0], 1, xs[2], 1)
        Xstd = Xstd.view(1, Xstd.size(0),1,1).repeat(xs[0], 1, xs[2], 1)
        z_x = (x - Xavg)/Xstd
        
        #z_x = self.bn(z_x)
        
        conv1 = self.conv1(z_x)
        b1, bnb1 = self.b1(conv1)
        mp1 = F.max_pool2d(b1, (1,4))
        b2, bnb2 = self.b2(mp1)
        mp2 = F.max_pool2d(b2, (1,4))
        b3, bnb3 = self.b3(mp2)
        bf = F.relu(self.bnf(b3))
        
        # pooling
        tpl = self.nn_apl(bf)

        #DNN
        den1 = F.relu(self.dbn1(self.den1(self.dp(tpl))))
        den2 = F.relu(self.dbn2(self.den2(self.dp(den1))))
        y_cls = self.prd(self.dp(den2))
            
        # attention
        att = self.nn_att(bf, self.att)
        det = self.det(bf)
        
        # ensemble prediction
        att_ens = F.softmax(att, dim=3)
        y_att = (det * att_ens).sum(-1).sum(-1)
        y_ens = (y_cls + y_att)/2
        
        return y_ens, [[att, det]]



