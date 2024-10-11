

# Anti-noise_FGVR

This repository provides a PyTorch implementation of the fine-grained vehicle recognition method, as proposed in my paper: [Progressive Multi-Task Anti-Noise Learning and Distilling Frameworks for Fine-Grained Vehicle Recognition](https://ieeexplore.ieee.org/document/10623841).

### Figure 1: The target problem addressed by the proposed method

![The target problem addressed by the proposed method.](https://raw.githubusercontent.com/Dichao-Liu/Anti-noise_FGVR/main/noise_problem.png)


### Figure 2: The proposed module

![The proposed module.](https://raw.githubusercontent.com/Dichao-Liu/Anti-noise_FGVR/main/DRH.png)







### Environment

This source code was tested in the following environment:

    Python = 3.8.13
    PyTorch = 1.12.0
    torchvision = 0.13.0
    Ubuntu 20.04.6 LTS
    NVIDIA GeForce RTX 3080 Ti

### Pre-trained Models
The pre-trained models can be downloaded from [this link](https://wani.teracloud.jp/share/11f23df41b4a6f82).

Please save the downloaded models in the `weightsFromCloud` folder.

The `xxxxx_Network.pth` file was saved using `torch.save(model, 'xxxxx_Network.pth')`.

The `xxxxx_Weight.pth` file was saved using `torch.save(model.state_dict(), 'xxxxx_Weight.pth')`.

If you decide to register on InfiniCLOUD to download the model, I would appreciate it if you could kindly use my referral code `XTQQJ` during the process. This small gesture will be of great help to me.


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
    
When training the student network, the `--from_local` option allows you to specify whether to use the teacher model downloaded from InfiniCLOUD or a model you have trained yourself using the provided code.


### Inference

    python Inference_Stanford_Cars_ResNet50_Teacher.py
    python Inference_Stanford_Cars_ResNet50_Student.py


### Bibtex

```
@ARTICLE{10623841,
  author={Liu, Dichao},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Progressive Multi-Task Anti-Noise Learning and Distilling Frameworks for Fine-Grained Vehicle Recognition}, 
  year={2024},
  volume={25},
  number={9},
  pages={10667-10678},
  keywords={Noise;Task analysis;Image recognition;Multitasking;Training;Noise measurement;Accuracy;Fine-grained vehicle recognition;intelligent transportation systems;ConvNets;object recognition},
  doi={10.1109/TITS.2024.3420151}
}
```





