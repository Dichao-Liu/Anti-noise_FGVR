

# Anti-noise_FGVR


### Environment

This source code was tested in the following environment:

    Python = 3.8.13
    PyTorch = 1.12.0
    torchvision = 0.13.0
    Ubuntu 20.04.6 LTS
    NVIDIA GeForce RTX 3080 Ti

### Pre-trained Models
The pre-trained models can be downloaded from https://wani.teracloud.jp/share/11f23df41b4a6f82.

`xxxxx_Network.pth` was saved by `torch.save(model, 'xxxxx_Network.pth')`.

`xxxxx_Weight.pth` was saved by `torch.save(model.state_dict(), 'xxxxx_Weight.pth')`.

### Dependencies

* **(1) Installation**

Install `Inplace-ABN` following the instructions:

https://github.com/Alibaba-MIIL/TResNet/blob/master/requirements.txt

https://github.com/Alibaba-MIIL/TResNet/blob/master/INPLACE_ABN_TIPS.md

Install `imgaug`:

    pip install imgaug

* **(2) Download**

Download the folder `src` from https://github.com/Alibaba-MIIL/TResNet,
the folder `vic` from https://github.com/styler00dollar/pytorch-loss-functions,
the folder `example` and Python file `sam.py` from https://github.com/davda54/sam,
and save them as:

    Anti-noise_FGVR
    ├── basic_conv.py
    ├── Inference_Stanford_Cars_ResNet50_Student.py
    ├── Inference_Stanford_Cars_ResNet50_Teacher.py
    ├── ...
    ├── src
    ├── vic
    ├── example
    ├── sam.py

Alternatively, you can simply download by running the following commands (note that `subversion` should be installed beforehand as `sudo apt install subversion`):

    git clone https://github.com/Dichao-Liu/Anti-noise_FGVR.git
    cd Anti-noise_FGVR
    svn export https://github.com/Alibaba-MIIL/TResNet/branches/master/src
    svn export https://github.com/davda54/sam/branches/main/example
    svn export https://github.com/davda54/sam/branches/main/sam.py
    svn export https://github.com/styler00dollar/pytorch-loss-functions/branches/main/vic


### Dataset

* **(1) Download the Stanford Cars dataset or other datasets mentioned in the paper, and organize the structure as follows:**
```
dataset folder
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── test
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```
* **(2) modify the path to the dataset folders.**

### Train

    python Stanford_Cars_ResNet50_PMAL.py
    python Stanford_Cars_ResNet50_Distillation.py

### Inference

    python Inference_Stanford_Cars_ResNet50_Teacher.py
    python Inference_Stanford_Cars_ResNet50_Student.py




