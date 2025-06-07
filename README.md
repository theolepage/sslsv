<p align="center">
  <img src="logo.png" width=180 />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Python-3.13-aff?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-2.7-blue?logo=pytorch">
</p>

# sslsv

**sslsv** is a PyTorch-based Deep Learning framework consisting of a collection of **Self-Supervised Learning** (SSL) methods for learning speaker representations applicable to different speaker-related downstream tasks, notably **Speaker Verification** (SV).

Our aim is to: **(1) provide self-supervised SOTA methods** by porting algorithms from the computer vision domain; and **(2) evaluate them in a comparable environment**.

Our training framework is depicted by the figure below.

<p align="center">
  <img src="training_framework.svg" width=900 />
</p>

---

## News

* **April 2024** – :clap: Introduction of new various methods and complete refactoring (v2.0).
* **June 2022** – :stars: First release of sslsv (v1.0).

---

## Features

**General**

- **Data**:
  - Supervised and Self-supervised datasets (siamese and DINO sampling)
  - Audio augmentation (noise and reverberation)
- **Training**:
  - CPU, GPU and multi-GPUs (*DataParallel* and *DistributedDataParallel*)
  - Checkpointing, resuming, early stopping and logging
  - Tensorboard and wandb
- **Evaluation**:
  - Speaker verification
    - Backend: Cosine scoring and PLDA
    - Metrics: EER, MinDCF, ActDFC, CLLR, AvgRPrec
  - Classification (emotion, language, ...)
- **Notebooks**: DET curve, scores distribution, t-SNE on embeddings, ...
- **Misc**: scalable config, typing, documentation and tests

<details>
  <summary><b>Encoders</b></summary>

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
  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification ([arXiv](https://arxiv.org/abs/2005.07143))  
  *Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck*
</details>

<details>
  <summary><b>Methods</b></summary>

- **LIM** (`sslsv.methods.LIM`)  
  Learning Speaker Representations with Mutual Information ([arXiv](https://arxiv.org/abs/1812.00271))  
  *Mirco Ravanelli, Yoshua Bengio*

- **CPC** (`sslsv.methods.CPC`)  
  Representation Learning with Contrastive Predictive Coding ([arXiv](https://arxiv.org/abs/1807.03748))  
  *Aaron van den Oord, Yazhe Li, Oriol Vinyals*

- **SimCLR** (`sslsv.methods.SimCLR`)  
  A Simple Framework for Contrastive Learning of Visual Representations ([arXiv](https://arxiv.org/abs/2002.05709))  
  *Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton*

- **MoCo v2+** (`sslsv.methods.MoCo`)  
  Improved Baselines with Momentum Contrastive Learning ([arXiv](https://arxiv.org/abs/2003.04297))  
  *Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He*

- **DeepCluster v2** (`sslsv.methods.DeepCluster`)  
  Deep Clustering for Unsupervised Learning of Visual Features ([arXiv](https://arxiv.org/abs/1807.05520))  
  *Mathilde Caron, Piotr Bojanowski, Armand Joulin, Matthijs Douze*

- **SwAV** (`sslsv.methods.SwAV`)  
  Unsupervised Learning of Visual Features by Contrasting Cluster Assignments ([arXiv](https://arxiv.org/abs/2006.09882))  
  *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin*

- **W-MSE** (`sslsv.methods.WMSE`)  
  Whitening for Self-Supervised Representation Learning ([arXiv](https://arxiv.org/abs/2007.06346))  
  *Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, Nicu Sebe*

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
</details>

<details open>
  <summary><b>Methods (ours)</b></summary>

- **Combiner** (`sslsv.methods.Combiner`)  
  Label-Efficient Self-Supervised Speaker Verification With Information Maximization and Contrastive Learning ([arXiv](https://arxiv.org/abs/2207.05506))  
  *Theo Lepage, Reda Dehak*

- **SimCLR Margins** (`sslsv.methods.SimCLRMargins`)  
  Additive Margin in Contrastive Self-Supervised Frameworks to Learn Discriminative Speaker Representations ([arXiv](https://arxiv.org/abs/2404.14913))  
  *Theo Lepage, Reda Dehak*

- **MoCo Margins** (`sslsv.methods.MoCoMargins`)  
  Additive Margin in Contrastive Self-Supervised Frameworks to Learn Discriminative Speaker Representations ([arXiv](https://arxiv.org/abs/2404.14913))  
  *Theo Lepage, Reda Dehak*

- **SSPS** (`sslsv.methods._SSPS`)  
  Self-Supervised Frameworks for Speaker Verification via Bootstrapped Positive Sampling
 ([arxiv](https://arxiv.org/abs/2501.17772))  
 *Theo Lepage, Reda Dehak*

</details>


---

## Requirements

*sslsv* runs on Python 3.13.3 with the following dependencies.

| Module                | Versions |
|-----------------------|:--------:|
| torch                 | 2.7.1    |
| torchaudio            | 2.7.1    |
| numpy                 | *        |
| pandas                | *        |
| soundfile             | *        |
| scikit-learn          | *        |
| speechbrain           | *        |
| tensorboard           | *        |
| wandb                 | *        |
| ruamel.yaml           | *        |
| dacite                | *        |
| prettyprinter         | *        |
| tqdm                  | *        |

**Note**: developers will also need `pytest`, `pre-commit` and `twine` to work on this project.

---

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

Data used for main experiments (conducted on VoxCeleb1 and VoxCeleb2 + data-augmentation) can be automatically downloaded, extracted and prepared using the following scripts.

```bash
python tools/prepare_data/prepare_voxceleb.py data/
python tools/prepare_data/prepare_augmentation.py data/
```

The resulting `data` folder shoud have the structure presented below.

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

Other datasets have to be manually downloaded and extracted but their train and trials files can be created using the corresponding scripts from the `tools/prepare_data/` folder.

- Example format of a train file (`voxceleb1_train.csv`)
  ```
  File,Speaker
  voxceleb1/id10001/1zcIwhmdeo4/00001.wav,id10001
  ...
  voxceleb1/id11251/s4R4hvqrhFw/00009.wav,id11251
  ```

- Example format of a trials file (`voxceleb1_test_O`)
  ```
  1 voxceleb1/id10270/x6uYqmx31kE/00001.wav voxceleb1/id10270/8jEAjG6SegY/00008.wav
  ...
  0 voxceleb1/id10309/0cYFdtyWVds/00005.wav voxceleb1/id10296/Y-qKARMSO7k/00001.wav
  ```

<!-- *Please refer to the associated code if you want further details about data preparation.* -->

---

## Installation

1. **Clone this repository**: `git clone https://github.com/theolepage/sslsv.git`.
2. **Install dependencies**: `pip install -r requirements.txt`.

**Note**: *sslsv* can also be installed as a standalone package via pip with `pip install sslsv` or with `pip install .` (in the project root folder) to get the latest version.

---

## Usage


- **Start a training** (*2 GPUs*): `./train_ddp.sh 2 <config_path>`.
- **Evaluate your model** (*2 GPUs*): `./evaluate_ddp.sh 2 <config_path>`.

**Note**: use `sslsv/bin/train.py` and `sslsv/bin/evaluate.py` for non-distributed mode to run with a CPU, a single GPU or multiple GPUs (*DataParallel*).


### Tensorboard

You can visualize your experiments with `tensorboard --logdir models/your_model/`.

### wandb

Use `wandb online` and `wandb offline` to toggle wandb. To log your experiments you first need to provide your API key with `wandb login API_KEY`.

---

## Documentation

*Documentation is currently being developed...*

---

## Results

### SOTA

- **Train set**: VoxCeleb2
- **Evaluation**: VoxCeleb1-O (Original)
- **Encoder**: ECAPA-TDNN (C=1024)

| Method         | Model                                              | EER (%) | minDCF (p=0.01)  | Checkpoint    |
|----------------|----------------------------------------------------|:-------:|:----------------:|:-------------:|
| **SimCLR**     | `ssl/voxceleb2/simclr/simclr_e-ecapa-1024`         | 6.41    | 0.5160           | [:link:](https://drive.google.com/drive/folders/1jQO5cYDUw5sEemkFPmrBXAaVIW943lE8?usp=sharing) |
| **MoCo**       | `ssl/voxceleb2/moco/moco_e-ecapa-1024`             | 6.38    | 0.5384           | [:link:](https://drive.google.com/drive/folders/1du3e0DaavfuN16kSqCXFSj1EM74MoGWa?usp=sharing) |
| **SwAV**       | `ssl/voxceleb2/swav/swav_e-ecapa-1024`             | 8.33    | 0.6120           | [:link:](https://drive.google.com/drive/folders/1ShF3qEzzw_eVJ9guP8isopl-cUwR-2Sn?usp=sharing) |
| **VICReg**     | `ssl/voxceleb2/vicreg/vicreg_e-ecapa-1024`         | 7.85    | 0.6004           | [:link:](https://drive.google.com/drive/folders/1_SIlJrMXk7G0inims3efFKNOwUMCZOC6?usp=sharing) |
| **DINO**       | `ssl/voxceleb2/dino/dino+_e-ecapa-1024`            | 2.92    | 0.3523           | [:link:](https://drive.google.com/drive/folders/1kQyJ_QdlneX_t9UgjZev6BpUO6zflcUR?usp=sharing) |
| **Supervised** | `ssl/voxceleb2/supervised/supervised_e-ecapa-1024` | 1.34    | 0.1521           | [:link:](https://drive.google.com/drive/folders/1vKTEdHBnVl_22838n4OGH_h8Nx--yl_P?usp=sharing) |

---

## Acknowledgements

*sslsv* contains third-party components and code adapted from other open-source projects, including: [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer), [voxceleb_unsupervised](https://github.com/joonson/voxceleb_unsupervised) and [solo-learn](https://github.com/vturrisi/solo-learn).

---

## Citations

If you use *sslsv*, please consider starring this repository on GitHub and citing one the following papers.

```BibTeX
@InProceedings{lepage2025SSPS,
  title     = {SSPS: Self-Supervised Positive Sampling for Robust Self-Supervised Speaker Verification},
  author    = {Lepage, Theo and Dehak, Reda},
  year      = {2025},
  booktitle = {Interspeech 2025},
  url       = {https://arxiv.org/abs/2505.14561},
}

@Article{lepage2025SSLSVBootstrappedPositiveSampling,
  title     = {Self-Supervised Frameworks for Speaker Verification via Bootstrapped Positive Sampling},
  author    = {Lepage, Theo and Dehak, Reda},
  year      = {2025},
  journal   = {arXiv preprint library},
  url       = {https://arxiv.org/abs/2501.17772},
}

@InProceedings{lepage2024AdditiveMarginSSLSV,
  title     = {Additive Margin in Contrastive Self-Supervised Frameworks to Learn Discriminative Speaker Representations},
  author    = {Lepage, Theo and Dehak, Reda},
  year      = {2024},
  booktitle = {The Speaker and Language Recognition Workshop (Odyssey 2024)},
  pages     = {38--42},
  doi       = {10.21437/odyssey.2024-6},
  url       = {https://www.isca-archive.org/odyssey_2024/lepage24_odyssey.html},
}

@InProceedings{lepage2023ExperimentingAdditiveMarginsSSLSV,
  title     = {Experimenting with Additive Margins for Contrastive Self-Supervised Speaker Verification},
  author    = {Lepage, Theo and Dehak, Reda},
  year      = {2023},
  booktitle = {Interspeech 2023},
  pages     = {4708--4712},
  doi       = {10.21437/Interspeech.2023-1479},
  url       = {https://www.isca-archive.org/interspeech_2023/lepage23_interspeech.html},
}

@InProceedings{lepage2022LabelEfficientSSLSV,
  title     = {Label-Efficient Self-Supervised Speaker Verification With Information Maximization and Contrastive Learning},
  author    = {Lepage, Theo and Dehak, Reda},
  year      = {2022},
  booktitle = {Interspeech 2022},
  pages     = {4018--4022},
  doi       = {10.21437/Interspeech.2022-802},
  url       = {https://www.isca-archive.org/interspeech_2022/lepage22_interspeech.html},
}
```

---

## License

This project is released under the [MIT License](https://github.com/theolepage/sslsv/blob/main/LICENSE.md).
