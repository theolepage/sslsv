#!/bin/bash

MODELS=(
    "models/ssps/voxceleb2/simclr_p-none/baseline/config.yml"
    "models/ssps/voxceleb2/simclr_p-none/baseline_sup/config.yml"
    "models/ssps/voxceleb2/simclr_p-none/ssps_kmeans_6k/config.yml"
    "models/ssps/voxceleb2/simclr_p-none/ssps_kmeans_25k_uni-1/config.yml"
    "models/ssps/voxceleb2/simclr_p-none/ssps_kmeans_50k_uni-1/config.yml"

    "models/ssps/voxceleb2/vicreg/baseline/config.yml"
    "models/ssps/voxceleb2/vicreg/baseline_sup/config.yml"
    "models/ssps/voxceleb2/vicreg/ssps_kmeans_6k/config.yml"
    "models/ssps/voxceleb2/vicreg/ssps_kmeans_25k_uni-1/config.yml"
    "models/ssps/voxceleb2/vicreg/ssps_kmeans_50k_uni-1/config.yml"

    "models/ssps/voxceleb2/swav/baseline/config.yml"
    "models/ssps/voxceleb2/swav/baseline_sup/config.yml"
    "models/ssps/voxceleb2/swav/ssps_kmeans_6k/config.yml"
    "models/ssps/voxceleb2/swav/ssps_kmeans_25k_uni-1/config.yml"
    "models/ssps/voxceleb2/swav/ssps_kmeans_50k_uni-1/config.yml"
)


for config in "${MODELS[@]}"; do
    torchrun --nproc_per_node=2 sslsv/bin/create_ssps_buffers_distributed.py "$config"

    ./train_ddp.sh 2 "$config"

    # ./evaluate_ddp.sh 2 "$config"
    python sslsv/bin/average_model.py "$config" --silent --count 5 --limit_nb_epochs 10
    torchrun --nproc_per_node=2 sslsv/bin/evaluate_distributed.py "$config" --model_suffix avg
done
