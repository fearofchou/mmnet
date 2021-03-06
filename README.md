# M&mnet (A CNN-based Network for Audio Recognition)
Pytorch implementation of [Learning to Recognize Transient Sound Events Using Attentional Supervision]


## Citation
If you use M&mnet in your research, please cite our paper

    @article{Chou2018mmnet,
      title={Learning to Recognize Transient Sound Events Using Attentional Supervision},
      author={Szu-Yu Chou and Jyh-Shing Roger Jang and Yi-Hsuan Yang},
      journal={in Proc. Int. Joint Conf. Artificial Intelligence (IJCAI)},
      year={2018}
    }

## Requirements
* Python 2.7
* LibROSA 0.6.0
* PyTorch 0.4.0
* cuda-8.0
* Download [Pytorch 2.7](https://pytorch.org)
```bash
pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl
pip install torchvision
```

## Download Task4_DCASE2017_dataset
* [Training Set](https://drive.google.com/file/d/1HOQaUHbTgCRsS6Sr9I9uE6uCjiNPC3d3/view)
* [Testing Set](https://drive.google.com/file/d/1GfP5JATSmCqD8p3CBIkk1J90mfJuPI-k/view)
* [Password](https://groups.google.com/forum/#!searchin/dcase-discussions/own%7Csort:relevance/dcase-discussions/Lk2dTScX3A8/kvW17tlzAgAJ)


## Usage
Note: You need to modify to your dataset path before you run the code.

	$ https://github.com/fearofchou/mmnet.git
    $ python main.py

## Pre-trained models
Model |DataSet |
:----:|:--------:|
[M&mnet](https://drive.google.com/file/d/1cdaQNltci_9namelgMS3Vjc16kF8g8A9/view?usp=sharing)|[DCASE2017-Task4](http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-large-scale-sound-event-detection)
[M&mnet (Coming soon)](https://github.com/fearofchou/mmnet)|[AudioSet-2M](https://research.google.com/audioset/)

