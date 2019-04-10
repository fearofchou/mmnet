import argparse
from Test import *

# pre_trained model path
#pmp = './Mmnet_DCASE17'
pmp = '../model/Mmnet_1D_MS_att/epoch_100'

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
pred_fn = '../pred/%s.npy'%(pmp.split('/')[2])

try:
    out = np.load(pred_fn)
except:
    out = Trer.predictor()
    np.save('../pred/%s'%(pmp.split('/')[2]), out)

vt = np.load('./tmp_va_the.npy')
music_mood = np.arange(276,283)
music_genre = np.arange(216,265)
music = np.append(music_mood, music_genre)

with open('/home/fearofchou/ND/m189/max/FCNN_torch/pre/csv/class_labels_indices.csv', 'r') as f:
    cl = f.readlines()

id2tag = {}
a = 0
for i in music:
    tag = cl[i+1].split(',')[-1][1:-2]
    id2tag[a] = tag
    a+=1


get_high_recall = 0.02
out_tag = {}
for i in xrange(len(out)):
    out_tag[Trer.fn[i]] = []
    for j in np.arange(len(id2tag))[out[i] > (vt-get_high_recall)]:
        out_tag[Trer.fn[i]].append(id2tag[j])
 
np.save('../out_tag/%s_%s'%(pmp.split('/')[2], str(get_high_recall)[2:]), out_tag)



