import torch
import torch.nn as nn
import torch.nn.functional as F

def l_tpl(x, attention=[]):
    # Global temproal pooling
    xs = x.size()
    # Global average pooling
    apl = F.avg_pool2d(x, xs[2:]).view(xs[0], -1)
    # Global max pooling
    mpl = F.max_pool2d(x, xs[2:]).view(xs[0], -1)
    
    # Global variance pooling
    var = (x - apl.view(xs[0], xs[1], 1, 1).repeat(1,1,xs[2],xs[3]) )**2
    vpl = F.avg_pool2d(var, xs[2:]).view(xs[0], -1)
    
    '''
    sm_att = F.softmax(attention.permute(0,2,1,3)).permute(0,2,1,3).repeat(1,xs[1],1,1)
    att = x * sm_att
    att = att.sum(dim=-1).sum(dim=-1)
    '''
    #return torch.cat([attention, mpl, vpl], dim=1)
    return torch.cat([apl, vpl, mpl], dim=1)
    #return att, apl, mpl, vpl

