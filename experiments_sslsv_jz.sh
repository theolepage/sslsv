#!/bin/bash



# ----------
# Supervised
# ----------

./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_aug-25/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_aug-50/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_aug-75/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_aug-none/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_train-half-spk/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_train-half-utt/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_train-quarter-spk/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2_train-quarter-utt/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-512_loss-AAM_s-30_m-0.2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-emotion-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-emotion-unfrozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-gender-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-gender-unfrozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-language-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-language-unfrozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-speaker-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2_classif-speaker-unfrozen/

torchrun --nproc_per_node=2 sslsv/bin/evaluate_label_efficient_distributed.py \
    models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2/config.yml



# ---
# LIM
# ---

./train_ddp_jz.sh 2 models/ssl/voxceleb2/lim/lim_loss-BCE_proj-none/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/lim/lim_loss-MINE_proj-none/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/lim/lim_loss-NCE_proj-none/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/lim/lim_loss-NCE_proj-2048-BN-R-2048-BN-R-512/



# ---
# CPC
# ---

./train_ddp_jz.sh 2 models/ssl/voxceleb2/cpc/cpc_t-4_agg-GRU-1-256/



# ------
# SimCLR
# ------

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-R-128_t-0.5/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-none_t-0.5/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-BN-R-2048-BN-R-512_t-0.5/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-BN-R-2048-BN-R-512_t-0.01/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-BN-R-2048-BN-R-512_t-0.03/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-BN-R-2048-BN-R-512_t-0.05/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-BN-R-2048-BN-R-512_t-0.07/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-none_t-0.03/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-none_t-0.03_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-none_t-0.03_sup2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-BN-R-2048-BN-R-512_t-0.03_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-2048-BN-R-2048-BN-R-512_t-0.03_sup2/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_proj-none_t-0.03_aug-none/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-emotion-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-emotion-unfrozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-gender-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-gender-unfrozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-language-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-language-unfrozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-speaker-frozen/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03_classif-speaker-unfrozen/



# ----
# MoCo
# ----

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-2048-R-128_Q-65536_t-0.2_m-0.999/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-65536_t-0.2_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-65536_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-2048-BN-R-2048-BN-R-512_Q-65536_t-0.2_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-2048-BN-R-2048-BN-R-512_Q-65536_t-0.03_m-0.999/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-256_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-1024_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-16384_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-2048-BN-R-2048-BN-R-512_Q-32768_t-0.03_m-0.999/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-256_nofn_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-1024_nofn_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-16384_nofn_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_nofn_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-65536_nofn_t-0.03_m-0.999/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.01_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.05_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.07_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.1_m-0.999/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.0/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.9/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.99/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.996/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-1.0/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.99-sched/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.996-sched/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999-sched/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_aug-none/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_aug-25/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_aug-50/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_aug-75/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_collapse-default/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_collapse-nonegs/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_collapse-hightemp/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_collapse-lowtemp/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_sup2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_sup3/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-2048-BN-R-2048-BN-R-512_Q-32768_t-0.03_m-0.999_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-2048-BN-R-2048-BN-R-512_Q-32768_t-0.03_m-0.999_sup2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-2048-BN-R-2048-BN-R-512_Q-32768_t-0.03_m-0.999_sup3/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_train-half-spk/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_train-half-utt/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_train-quarter-spk/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999_train-quarter-utt/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_enc-ECAPATDNN-1024_proj-none_Q-65536_t-0.03_m-0.999/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/moco/moco_enc-ECAPATDNN-1024_proj-none_Q-32768_t-0.03_m-0.999/



# -----------
# DeepCluster
# -----------

./train_ddp_jz.sh 2 models/ssl/voxceleb2/deepcluster/deepcluster_proj-2048-BN-R-128_K-3000-3000-3000_t-0.1/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/deepcluster/deepcluster_proj-2048-BN-R-2048-BN-R-512_K-3000-3000-3000_t-0.1/



# ----
# SwAV
# ----

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-128_K-3000_t-0.1/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-none_K-3000_t-0.1/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-3000_t-0.1/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-1000_t-0.1/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-9000_t-0.1/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-12000_t-0.1/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.01/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.05/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.3/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-none_K-6000_t-0.1/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1_aug-none/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-none_K-6000_t-0.1_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-none_K-6000_t-0.1_sup2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-none_K-6000_t-0.1_sup3/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1_sup2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1_sup3/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_enc-ECAPATDNN-1024_proj-2048-BN-R-2048-BN-R-512_K-3000_t-0.1/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/swav/swav_enc-ECAPATDNN-1024_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1/



# -----
# W-MSE
# -----

./train_ddp_jz.sh 2 models/ssl/voxceleb2/wmse/wmse_proj-1024-BN-R-64_ws-128/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/wmse/wmse_proj-2048-BN-R-2048-BN-R-64_ws-128/



# ------------
# Barlow Twins
# ------------

./train_ddp_jz.sh 2 models/ssl/voxceleb2/barlowtwins/barlowtwins_proj-8192-BN-R-8192-BN-R-8192_lambda-0.005/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/barlowtwins/barlowtwins_proj-2048-BN-R-2048-BN-R-512_lambda-0.005/



# ------
# VICReg
# ------

./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-8192-BN-R-8192-BN-R-8192_inv-1.0_var-1.0_cov-0.04/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-none_inv-1.0_var-1.0_cov-0.04/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.04/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.0/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-0.5_cov-0.1/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-none_inv-1.0_var-1.0_cov-0.1/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1_aug-none/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-none_inv-1.0_var-1.0_cov-0.1_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-none_inv-1.0_var-1.0_cov-0.1_sup2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-none_inv-1.0_var-1.0_cov-0.1_sup3/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1_sup/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1_sup2/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1_sup3/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_enc-ECAPATDNN-1024_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.04/
./train_ddp_jz.sh 2 models/ssl/voxceleb2/vicreg/vicreg_enc-ECAPATDNN-1024_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1/



# ----
# BYOL
# ----

./train_ddp_jz.sh 2 models/ssl/voxceleb2/byol/byol_proj-4096-BN-R-256_pred-4096-BN-R-256_m-0.996-sched/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/byol/byol_proj-2048-BN-R-2048-BN-R-512_pred-4096-BN-R-256_m-0.996-sched/



# -------
# SimSiam
# -------

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simsiam/simsiam_proj-2048-BN-R-2048-BN-R-2048-BN_pred-512-BN-R-2048/

./train_ddp_jz.sh 2 models/ssl/voxceleb2/simsiam/simsiam_proj-2048-BN-R-2048-BN-R-512-BN_pred-512-BN-R-2048/



# ----
# DINO
# ----

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_setup_base_+wd/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_setup_base_+wd_+lr-sched/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_setup_base_+wd_+lr-sched_+pooling-stats/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_setup_base_+wd_+lr-sched_+pooling-stats_+aug-all/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_setup_base_+wd_+lr-sched_+pooling-stats_+aug-all_+input-80d/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_setup_base_+wd_+lr-sched_+pooling-stats_+aug-all_+input-80d_+optim-sgd_+mel-fn-hann/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-1x2_L-1x2_t-0.04/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x3_L-4x2_t-0.04/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x2_L-4x2_t-0.04/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-1x2_L-4x2_t-0.04/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-2x2_t-0.04/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-2048_G-2x4_L-4x2_t-0.04/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-16384_G-2x4_L-4x2_t-0.04/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-32768_G-2x4_L-4x2_t-0.04/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.01/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.03/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.05/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.07/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04-0.07/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-L2-65536_G-2x4_L-4x2_t-0.04/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_aug-none/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_aug-25/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_aug-50/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_aug-75/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_collapse-default/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_collapse-nocentering/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_collapse-nosharpening/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_train-half-spk/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_train-half-utt/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_train-quarter-spk/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_train-quarter-utt/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-L2-65536_G-2x4_L-4x2_t-0.04_sup/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-L2-65536_G-2x4_L-4x2_t-0.04_sup2/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-L2-65536_G-2x4_L-4x2_t-0.04_sup3/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_sup/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_sup2/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_sup3/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/

./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-emotion-frozen/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-emotion-unfrozen/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-gender-frozen/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-gender-unfrozen/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-language-frozen/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-language-unfrozen/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-speaker-frozen/
./train_ddp_jz.sh 4 models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04_classif-speaker-unfrozen/

torchrun --nproc_per_node=2 sslsv/bin/evaluate_label_efficient_distributed.py \
    models/ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2/config.yml \
    --encoder_config models/ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04/config.yml