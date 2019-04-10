import argparse
from Trainer import *

# pre_trained model path
#pmp = './Mmnet_DCASE17'
pmp = ''

# params for audio feature extraction (mel-spectrogram)
parser = argparse.ArgumentParser(description= 'PyTorch M&ment Training using AudioSet Dataset')
parser.add_argument('--dn',  default='AudioSet_Music', type=str, help='dataset name')
parser.add_argument('--sr',  default=44100, type=int, help='[fea_ext] sample rate')
parser.add_argument('--ws',  default=2048,  type=int, help='[fea_ext] windows size')
parser.add_argument('--hs',  default=512,   type=int, help='[fea_ext] hop size')
parser.add_argument('--mel', default=128,   type=int, help='[fea_ext] mel bands')
parser.add_argument('--msc', default=10,    type=int, help='[fea_ext] top duration of audio clip')

# params for training
parser.add_argument('--bs',   default=64,    type=int,   help='[net] batch size')
parser.add_argument('--gw',   default=1,     type=int,   help='[net] global weight for both positive and negative samples')
parser.add_argument('--lrde', default=30,    type=int,   help='[net] divided the learning rate 10 by every lrde epochs')
parser.add_argument('--mom',  default=0.9,   type=float, help='[net] momentum')
parser.add_argument('--wd',   default=1e-4,  type=float, help='[net] weight decay')
parser.add_argument('--lr',   default=0.1,   type=float, help='[net] learning rate')
parser.add_argument('--ep',   default=100,   type=int,   help='[net] epoch')
parser.add_argument('--beta', default=0.3,   type=float, help='[net] hyperparameter for pre-class loss weight')
parser.add_argument('--pmp',  default=pmp,   type=str,   help='[net] pre-trained model path')
args = parser.parse_args()


# build model
#model = nn.DataParallel(Net(args.mel, data['Ytr'].shape[1]).cuda())

# Train
Trer = Trainer(args)
Trer.fit()



