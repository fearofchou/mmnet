# A M&mnet-based Network for Music Auto-tagging)

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


## Usage
Note: You need to modify to your dataset path before you run the code.

	$ python ./pre/gen_ASmusic_data.py
    $ python ./main/main.py

## Pre-trained models
Model |Dataset |
:----:|:--------:|
[M&mnet](https://drive.google.com/file/d/1hfNTgH4WM2qlgIKrqizxqWNp7UvWvFBs/view?usp=sharing)|[AudioSetMusic](https://research.google.com/audioset/ontology/music_1.html)

