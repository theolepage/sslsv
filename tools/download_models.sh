#!/bin/bash

base_path=/lustre/fshomisc/home/rech/genoub01/uxu84ci/sslsv/./models

models=(
    "ssl/voxceleb2/simclr/simclr_e-ecapa-1024"
    "ssl/voxceleb2/moco/moco_e-ecapa-1024"
    "ssl/voxceleb2/vicreg/vicreg_e-ecapa-1024"
    "ssl/voxceleb2/swav/swav_e-ecapa-1024"
    "ssl/voxceleb2/dino/dino+_e-ecapa-1024"
    "ssl/voxceleb2/supervised/supervised_e-ecapa-1024"
)

for model in "${models[@]}"; do
    rsync -av --relative \
        jeanzay:${base_path}/${model}/{config.yml,checkpoints/model_latest.pt} .
done