# sslsv

Collection of **self-supervised learning** (SSL) methods for **speaker verification** (SV).

## Methods

### Encoders

- **TDNN** (`sslsv.encoders.TDNN`)  
  X-vectors: Robust dnn embeddings for speaker recognition ([PDF](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf))  
  *David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, Sanjeev Khudanpur*

- **Simple Audio CNN** (`sslsv.encoders.SimpleAudioCNN`)  
  Representation Learning with Contrastive Predictive Coding ([arXiv](https://arxiv.org/abs/1807.03748))  
  *Aaron van den Oord, Yazhe Li, Oriol Vinyals*

- **ResNet-34** (`sslsv.encoders.ResNet34`)  
  VoxCeleb2: Deep Speaker Recognition ([arXiv](https://arxiv.org/abs/1806.05622))  
  *Joon Son Chung, Arsha Nagrani, Andrew Zisserman*

- **ECAPA-TDNN** (`sslsv.encoders.ECAPATDNN`)  
  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification ([PDF](https://arxiv.org/abs/2005.07143))  
  *Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck*

### Methods

- **CPC** (`sslsv.methods.CPC`)  
  Representation Learning with Contrastive Predictive Coding ([arXiv](https://arxiv.org/abs/1807.03748))  
  *Aaron van den Oord, Yazhe Li, Oriol Vinyals*

- **LIM** (`sslsv.methods.LIM`)  
  Learning Speaker Representations with Mutual Information ([arXiv](https://arxiv.org/abs/1812.00271))  
  *Mirco Ravanelli, Yoshua Bengio*

- **SimCLR** (`sslsv.methods.SimCLR`)  
  A Simple Framework for Contrastive Learning of Visual Representations ([arXiv](https://arxiv.org/abs/2002.05709))  
  *Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton*

- **MoCo v2+** (`sslsv.methods.MoCo`)  
  Improved Baselines with Momentum Contrastive Learning ([arXiv](https://arxiv.org/abs/2003.04297))  
  *Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He*

- **Barlow Twins** (`sslsv.methods.BarlowTwins`)  
  Barlow Twins: Self-Supervised Learning via Redundancy Reduction ([arXiv](https://arxiv.org/abs/2103.03230))  
  *Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny*

- **VICReg** (`sslsv.methods.VICReg`)  
  VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning ([arXiv](https://arxiv.org/abs/2105.04906))  
  *Adrien Bardes, Jean Ponce, Yann LeCun*

- **VIbCReg** (`sslsv.methods.VIbCReg`)  
  Computer Vision Self-supervised Learning Methods on Time Series ([arXiv](https://arxiv.org/abs/2109.00783))  
  *Daesoo Lee, Erlend Aune*

- **BYOL** (`sslsv.methods.BYOL`)  
  Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning ([arXiv](https://arxiv.org/abs/2006.07733))  
  *Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko*

- **SimSiam** (`sslsv.methods.SimSiam`)  
  Exploring Simple Siamese Representation Learning ([arXiv](https://arxiv.org/abs/2011.10566))  
  *Xinlei Chen, Kaiming He*

- **DINO** (`sslsv.methods.DINO`)  
  Emerging Properties in Self-Supervised Vision Transformers ([arXiv](https://arxiv.org/abs/2104.14294))  
  *Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin*

- **DeepCluster v2** (`sslsv.methods.DeepCluster`)  
  Deep Clustering for Unsupervised Learning of Visual Features ([arXiv](https://arxiv.org/abs/1807.05520))  
  *Mathilde Caron, Piotr Bojanowski, Armand Joulin, Matthijs Douze*

- **SwAV** (`sslsv.methods.SwAV`)  
  Unsupervised Learning of Visual Features by Contrasting Cluster Assignments ([arXiv](https://arxiv.org/abs/2006.09882))  
  *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin*

## Datasets

**Speaker recognition**:
- [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) (train and test)
- [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) (train)
- [SITW](http://www.speech.sri.com/projects/sitw/) (test)
- [VOiCES](https://iqtlabs.github.io/voices/) (test)

**Language recognition**:
- [VoxLingua107](https://bark.phon.ioc.ee/voxlingua107/)

**Emotion recognition**:
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

**Data-augmentation**:
- [MUSAN](http://www.openslr.org/17/)
- [Room Impulse Response and Noise Database](https://www.openslr.org/28/)

Data used for main experiments (conducted on VoxCeleb1 and VoxCeleb2 + data-augmentation) can be automatically downloaded, extracted and prepared using `utils/prepare_voxceleb.py` and `utils/prepare_augmentation.py`. The resulting `data` folder shoud have the following structure:

```
data
├── musan_split/
├── simulated_rirs/
├── voxceleb1/
├── voxceleb2/
├── voxceleb1_test_O
├── voxceleb1_test_H
├── voxceleb1_test_E
├── voxsrc2021_val
├── voxceleb1_train.csv
└── voxceleb2_train.csv
```

Other datasets have to be manually downloaded and extracted but their train and trials *(only for speaker verification)* files can be created using the corresponding script from the `utils` folder.

<details>
  <summary>Example format of a train file</summary>
  `voxceleb1_train.csv`
  ```
  File,Speaker
  voxceleb1/id10001/1zcIwhmdeo4/00001.wav,id10001
  ...
  voxceleb1/id11251/s4R4hvqrhFw/00009.wav,id11251
  ```
</details>

<details>
  <summary>Example format of a trials file</summary>
  `voxceleb1_test_O`
  ```
  1 voxceleb1/id10270/x6uYqmx31kE/00001.wav voxceleb1/id10270/8jEAjG6SegY/00008.wav
  ...
  0 voxceleb1/id10309/0cYFdtyWVds/00005.wav voxceleb1/id10296/Y-qKARMSO7k/00001.wav
  ```
</details>

*Please refer to the associated code if you want further details about data preparation.*

## Usage

Start self-supervised training with `python train.py configs/vicreg.yml`.

### wandb

Use `wandb online` and `wandb offline` to toggle wandb. To log your experiments you first need to provide your API key with `wandb login API_KEY`.

## Credits

Some parts of the code (data preparation, data augmentation and model evaluation) were adapted from [VoxCeleb trainer](https://github.com/clovaai/voxceleb_trainer) repository.