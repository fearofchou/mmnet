from data_generator import *
import argparse
from Mmnet import *
from Trainer import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# DCASE metadata file path
classes_fn = './data/sound_event_list_17_classes.txt'
training_set_csv_fn = './data/groundtruth_weak_label_training_set.csv'
test_set_csv_fn = './data/groundtruth_weak_label_testing_set.csv'

# DCASE wave file path
training_set_wav_fp = '/home/fearofchou/ND/m189/max/dataset/DCA17_4/Task_4_DCASE_2017_training_set/unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads'
test_set_wav_fp = '/home/fearofchou/ND/m189/max/dataset/DCA17_4/Task_4_DCASE_2017_testing_set/unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads'

# pre_trained model path
pmp = './data/model/Mmnet_1D_e_35'
#pmp = ''
# params for audio feature extraction (mel-spectrogram)
parser = argparse.ArgumentParser(description= 'PyTorch M&ment Training used DCASE2017 Dataset')
parser.add_argument('--dn',  default='DCASE17_task4', type=str, help='dataset name')
parser.add_argument('--sr',  default=44100, type=int, help='[fea_ext] sample rate')
parser.add_argument('--ws',  default=2048,  type=int, help='[fea_ext] windows size')
parser.add_argument('--hs',  default=492,   type=int, help='[fea_ext] hop size')
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

# Read (if it exist) or generate data for training
h5_fn = './data/%s_%d_%d_%d_%d.h5'%(args.dn, args.sr, args.ws, args.hs, args.mel)
h5_fn = '/home/fearofchou/%s_%d_%d_%d_%d.h5'%(args.dn, args.sr, args.ws, args.hs, args.mel)
data = get_h5_dataset(h5_fn, training_set_csv_fn, test_set_csv_fn, 
        training_set_wav_fp, test_set_wav_fp, classes_fn, args)


# build model
model = nn.DataParallel(Net(args.mel, data['Ytr'].shape[1]).cuda())

# Train
Trer = Trainer(data, model, args)
Trer.fit()



