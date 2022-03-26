# sslsv

Collection of **self-supervised learning** (SSL) methods for **speaker verification** (SV).

## Resources

### Models

- **`sslsv.model.ThinResNet34`**  
  "Delving into VoxCeleb: environment invariant speaker recognition" ([arxiv](https://arxiv.org/abs/1910.11238))  
  *Joon Son Chung, Jaesung Huh, Seongkyu Mun*

### Losses

- **`sslsv.losses.InfoNCE`**  
  "Representation Learning with Contrastive Predictive Coding" ([arxiv](https://arxiv.org/pdf/1807.03748.pdf))  
  *Aaron van den Oord, Yazhe Li, Oriol Vinyals*

- **`sslsv.losses.VICReg`**  
  "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" ([arxiv](https://arxiv.org/abs/2105.04906))  
  *Adrien Bardes, Jean Ponce, Yann LeCun*

- **`sslsv.losses.BarlowTwins`**  
  "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" ([arxiv](https://arxiv.org/abs/2103.03230))  
  *Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny*

## Datasets

[VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) are used for our experiments and we rely on [MUSAN](http://www.openslr.org/17/) and [Room Impulse Response and Noise Database](https://www.openslr.org/28/) for data augmentation.

To download, extract and prepare all datasets run `python prepare_data.py data/`.  The `data/` directory will have the structure detailed below.

```
data
├── musan_split/
├── simulated_rirs/
├── voxceleb1/
├── voxceleb2/
├── trials
├── voxceleb1_train_list
└── voxceleb2_train_list
```

Trials and train lists files are also automatically created with the following formats.

- `trials`
    ```
    1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav
    ...
    0 id10309/0cYFdtyWVds/00005.wav id10296/Y-qKARMSO7k/00001.wav
    ```

- `voxceleb1_train_list` and `voxceleb2_train_list`
    ```
    id00012 voxceleb2/id00012/21Uxsk56VDQ/00001.wav
    ...
    id09272 voxceleb2/id09272/u7VNkYraCw0/00027.wav
    ```

*Please refer to `prepare_data.py` script if you want further details about data preparation.*

## Usage

Start self-supervised training with `python train.py configs/vicreg_b256.yml`.

## To-Do

- [ ] DDP: adapt losses and supervised sampler
- [ ] Refactor evaluation (use AudioDataset class for handling test data)
- [ ] Documentation, comments, typing

## Credits

Some parts of the code (data preparation, data augmentation and model evaluation) were adapted from [VoxCeleb trainer](https://github.com/clovaai/voxceleb_trainer) repository.