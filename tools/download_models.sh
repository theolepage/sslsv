#!/bin/bash

base_path=/lustre/fshomisc/home/rech/genoub01/uxu84ci/sslsv/./models

models=(
    "ssl/voxceleb2/simclr/simclr_proj-none_t-0.03"
    "ssl/voxceleb2/moco/moco_proj-none_Q-32768_t-0.03_m-0.999"
    "ssl/voxceleb2/swav/swav_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1"
    "ssl/voxceleb2/vicreg/vicreg_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1"
    "ssl/voxceleb2/dino/dino_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04"
    "ssl/voxceleb2/supervised/supervised_loss-AAM_s-30_m-0.2"

    "ssl/voxceleb2/simclr/simclr_enc-ECAPATDNN-1024_proj-none_t-0.03"
    "ssl/voxceleb2/moco/moco_enc-ECAPATDNN-1024_proj-none_Q-32768_t-0.03_m-0.999"
    "ssl/voxceleb2/swav/swav_enc-ECAPATDNN-1024_proj-2048-BN-R-2048-BN-R-512_K-6000_t-0.1"
    "ssl/voxceleb2/vicreg/vicreg_enc-ECAPATDNN-1024_proj-2048-BN-R-2048-BN-R-512_inv-1.0_var-1.0_cov-0.1"
    "ssl/voxceleb2/dino/dino_enc-ECAPATDNN-1024_proj-2048-BN-G-2048-BN-G-256-L2-65536_G-2x4_L-4x2_t-0.04"
    "ssl/voxceleb2/supervised/supervised_enc-ECAPATDNN-1024_loss-AAM_s-30_m-0.2"

    "ssps/voxceleb2/simclr_e-ecapa/ssps_kmeans_25k_uni-1"
    "ssps/voxceleb2/dino_e-ecapa/ssps_kmeans_25k_uni-1"
)

for model in "${models[@]}"; do
    rsync -av --relative \
        jeanzay:${base_path}/${model}/{config.yml,checkpoints/model_avg.pt} .
done

# rsync -av --relative models/ bititan:~/theo/sslsv/models