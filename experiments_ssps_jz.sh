#!/bin/bash

# SimCLR (Fast ResNet-34)
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/baseline/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/baseline_sup/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/ssps_kmeans_6k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/ssps_kmeans_25k_uni-1/

# VICReg (Fast ResNet-34)
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/vicreg/baseline/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/vicreg/baseline_sup/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/vicreg/ssps_kmeans_6k/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/vicreg/ssps_kmeans_25k_uni-1/

# SwAV (Fast ResNet-34)
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/swav/baseline/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/swav/baseline_sup/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/swav/ssps_kmeans_6k/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/swav/ssps_kmeans_25k_uni-1/



# SimCLR (ECAPA-TDNN)
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/simclr_e-ecapa/baseline/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/simclr_e-ecapa/baseline_sup/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/simclr_e-ecapa/ssps_kmeans_6k/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/simclr_e-ecapa/ssps_kmeans_25k_uni-1/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/simclr_e-ecapa/ssps_kmeans_25k_uni-1_nofn/

# SwAV (ECAPA-TDNN)
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/swav_e-ecapa/baseline/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/swav_e-ecapa/baseline_sup/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/swav_e-ecapa/ssps_kmeans_6k/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/swav_e-ecapa/ssps_kmeans_25k_uni-1/

# VICReg (ECAPA-TDNN)
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/vicreg_e-ecapa/baseline/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/vicreg_e-ecapa/baseline_sup/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/vicreg_e-ecapa/ssps_kmeans_6k/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/vicreg_e-ecapa/ssps_kmeans_25k_uni-1/

# DINO (ECAPA-TDNN)
# ✅ ./train_ddp_ssps_jz_x4.sh models/ssps/voxceleb2/dino_e-ecapa/baseline/
# ✅ ./train_ddp_ssps_jz_x4.sh models/ssps/voxceleb2/dino_e-ecapa/baseline_sup/
# ✅ ./train_ddp_ssps_jz_x4.sh models/ssps/voxceleb2/dino_e-ecapa/ssps_kmeans_6k/
# ✅ ./train_ddp_ssps_jz_x4.sh models/ssps/voxceleb2/dino_e-ecapa/ssps_kmeans_25k_uni-1/

# MoCo (ECAPA-TDNN)
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/moco_e-ecapa/baseline/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/moco_e-ecapa/baseline_sup/
# ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/moco_e-ecapa/ssps_kmeans_6k/
# ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/moco_e-ecapa/ssps_kmeans_25k_uni-1/



# k-nn: M
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-1/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-10/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-25/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-50/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-100/

# k-means: K, M
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_6k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_10k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k_uni-1/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k_uni-2/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k_uni-3/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k_uni-5/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_50k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_50k_uni-1/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_50k_uni-2/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_50k_uni-3/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_50k_uni-5/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_100k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_100k_uni-1/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_100k_uni-2/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_100k_uni-3/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_100k_uni-5/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_150k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_150k_uni-1/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_150k_uni-2/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_150k_uni-3/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans_150k_uni-5/



# Repr/Centroid
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans-centroid_6k/
# ✅ ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_kmeans-centroid_25k_uni-1/



# Ref frame
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_frame-2s-clean/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_frame-3s-clean/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr/exps/ssps_frame-2s-aug/



# Data-aug
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/simclr_e-ecapa_v2/exps/ssps_aug-none/
# ✅ ./train_ddp_ssps_jz_x2.sh models/ssps/voxceleb2/simclr_e-ecapa_v2/exps/baseline_aug-none/



# Margins
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_curricularface_m-0.05/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_curricularface_m-0.1/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_curricularface_m-0.2/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_adaface_m-0.05/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_adaface_m-0.1/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_adaface_m-0.2/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_arcface_m-0.05/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_arcface_m-0.1/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_arcface_m-0.2/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_cosface_m-0.05/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_cosface_m-0.1/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_cosface_m-0.2/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_magface/
# ./train_ddp_ssps_jz_x2_exp.sh models/ssps/voxceleb2/simclr_margins/ssps_sphereface_m-0.1/
