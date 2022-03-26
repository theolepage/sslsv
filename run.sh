#!/bin/env sh

# mkdir data
# python prepare_data.py data

# Basic trainings
python train.py configs/infonce_b256.yml
python train.py configs/vicreg_b256.yml

# Data-augmentation
python train.py configs/vicreg_b256_aug_none.yml
python train.py configs/vicreg_b256_aug_musan.yml
python train.py configs/vicreg_b256_aug_rir.yml

# Batch size
python train.py configs/vicreg_b32.yml
python train.py configs/vicreg_b64.yml
python train.py configs/vicreg_b128.yml

# VICReg coefficients
python train.py configs/vicreg_b256_1_0.5_0.1.yml
python train.py configs/vicreg_b256_1_1_0.1.yml
python train.py configs/vicreg_b256_1_1_0.yml

# Projector dim
python train.py configs/vicreg_b256_mlp_none.yml
python train.py configs/vicreg_b256_mlp_512.yml
python train.py configs/vicreg_b256_mlp_1024.yml

# Complementarity
python train.py configs/vicreg_b256_comp_1.yml
python train.py configs/vicreg_b256_comp_2.yml
python train.py configs/vicreg_b256_comp_3.yml
python train.py configs/vicreg_b256_comp_4.yml

