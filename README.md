<p align="center">
  <img src="https://i.postimg.cc/CLFZLW7k/sslsv-logo-3-1.png" width=130 />
</p>

<!-- <p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/badge/Python-3.8-aff?logo=python">
  <img src="https://img.shields.io/badge/PyTorch-1.11.0-blue?logo=pytorch">
</p> -->

# sslsv

**sslsv** is a PyTorch-based Deep Learning framework consisting of a collection of **Self-Supervised Learning** (SSL) methods for learning speaker representations applicable to different speaker-related downstream tasks, notably **Speaker Verification** (SV).

Our aim is to: **(1) provide self-supervised SOTA methods** by porting algorithms from the computer vision domain; and **(2) evaluate them in a comparable environment**.

---

## News

* **April 2024** – :clap: Introduction of new various methods and complete refactoring (v2.0).
* **June 2022** – :stars: First release of sslsv (v1.0).

---

## Features

**General**

- **Data**: supervised and self-supervised datasets + augmentation (noise and reverberation)
- **Training**: CPU / multi-GPU (DP and DDP), resuming, early stopping, tensorboard, wandb, ...
- **Evaluation**: speaker verification (cosine and PLDA) and classification (emotion, language, ...)
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
  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification ([PDF](https://arxiv.org/abs/2005.07143))  
  *Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck*
</details>

<details>
  <summary><b>Methods</b></summary>

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

- **DeepCluster v2** (`sslsv.methods.DeepCluster`)  
  Deep Clustering for Unsupervised Learning of Visual Features ([arXiv](https://arxiv.org/abs/1807.05520))  
  *Mathilde Caron, Piotr Bojanowski, Armand Joulin, Matthijs Douze*

- **SwAV** (`sslsv.methods.SwAV`)  
  Unsupervised Learning of Visual Features by Contrasting Cluster Assignments ([arXiv](https://arxiv.org/abs/2006.09882))  
  *Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin*
</details>

<details open>
  <summary><b>Methods (ours)</b></summary>

- **Combiner** (`sslsv.methods.Combiner`)  
  Label-Efficient Self-Supervised Speaker Verification With Information Maximization and Contrastive Learning ([arXiv](https://arxiv.org/abs/2207.05506))  
  *Théo Lepage, Réda Dehak*

- **SimCLR Custom** (`sslsv.methods.SimCLRCustom`)  
  Experimenting with Additive Margins for Contrastive Self-Supervised Speaker Verification ([arXiv](https://arxiv.org/abs/2306.03664))  
  *Théo Lepage, Réda Dehak*

</details>


---

## Requirements

sslsv runs on Python 3.8 with the following dependencies.

| Module                | Versions  |
|-----------------------|:---------:|
| torch                 | >= 1.11.0 |
| torchaudio            | >= 0.11.0 |
| numpy                 | *         |
| pandas                | *         |
| soundfile             | *         |
| scikit-learn          | *         |
| speechbrain           | *         |
| tensorboard           | *         |
| wandb                 | *         |
| ruamel.yaml           | *         |
| dacite                | *         |
| prettyprinter         | *         |
| tqdm                  | *         |

**Note**: developers will also need `pre-commit` and `twine` to work on this project.

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

## Usage

1. **Clone the repository**: `git clone https://github.com/theolepage/sslsv.git`.
2. **Install dependencies**: `pip install -r requirements.txt`.
3. **Start a training** (*2 GPUs*): `./train_ddp.sh 2 <config_path>`.
4. **Evaluate your model** (*2 GPUs*): `./evaluate_ddp.sh 2 <config_path>`.

**Note 1**: with a CPU or a single GPU you can use `sslsv/bin/train.py` and `sslsv/bin/evaluate.py`, respectively.

**Note 2**: alternatively you can install sslsv using `pip install .` and use its modules separately from your code.

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
- **Evaluation**: SV on VoxCeleb1-O (original) trials
- **Encoder**: Fast ResNet-34

| Method      |              Model            | EER (%) | minDCF (p=0.01)  | Checkpoint    |
|-------------|:-----------------------------:|:-------:|:----------------:|:-------------:|
| **SimCLR**  | `ssl/voxceleb2/simclr/simclr` | -       | -                | [:link:](...) |
| ...         |                               |         |                  |               |

---

## Acknowledgements

sslsv contains third-party components and code adapted from other open-source projects, including: [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer), [voxceleb_unsupervised](https://github.com/joonson/voxceleb_unsupervised) and [solo-learn](https://github.com/vturrisi/solo-learn).

---

## Citations

If you use sslsv, please consider starring this repository on GitHub and citing one the following papers.

```BibTeX
@InProceedings{lepage2023ExperimentingAdditiveMarginsSSLSV,
  author    = {Lepage, Théo and Dehak, Réda},
  booktitle = {INTERSPEECH},
  title     = {Experimenting with Additive Margins for Contrastive Self-Supervised Speaker Verification},
  year      = {2023},
  url       = {https://www.isca-speech.org/archive/interspeech_2023/lepage23_interspeech.html},
}

@InProceedings{lepage2022LabelEfficientSelfSupervisedSV,
  author    = {Lepage, Théo and Dehak, Réda},
  booktitle = {INTERSPEECH},
  title     = {Label-Efficient Self-Supervised Speaker Verification With Information Maximization and Contrastive Learning},
  year      = {2022},
  url       = {https://www.isca-speech.org/archive/interspeech_2022/lepage22_interspeech.html},
}
```

---

## License

This project is released under the [MIT License](https://github.com/theolepage/sslsv/blob/main/LICENSE.md).
