#!/bin/bash

MODELS=(
    "models/ssps/voxceleb2/simclr_margins/ssps_sphereface_m-0.1/config.yml"
)


for config in "${MODELS[@]}"; do
    ./train_ddp.sh 2 "$config"
    ./evaluate_ddp.sh 2 "$config"
done
