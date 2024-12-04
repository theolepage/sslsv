#!/bin/bash

MODELS=(
    # Baselines
    "models/ssps/voxceleb2/simclr/baseline/config.yml"
    "models/ssps/voxceleb2/simclr/baseline_sup/config.yml"
    "models/ssps/voxceleb2/simclr/ssps_kmeans_6k/config.yml"
    "models/ssps/voxceleb2/simclr/ssps_kmeans_25k_uni-1/config.yml"

    "models/ssps/voxceleb2/vicreg/baseline/config.yml"
    "models/ssps/voxceleb2/vicreg/baseline_sup/config.yml"
    "models/ssps/voxceleb2/vicreg/ssps_kmeans_6k/config.yml"
    "models/ssps/voxceleb2/vicreg/ssps_kmeans_25k_uni-1/config.yml"

    "models/ssps/voxceleb2/swav/baseline/config.yml"
    "models/ssps/voxceleb2/swav/baseline_sup/config.yml"
    "models/ssps/voxceleb2/swav/ssps_kmeans_6k/config.yml"
    "models/ssps/voxceleb2/swav/ssps_kmeans_25k_uni-1/config.yml"

    # Data-aug
    "models/ssps/voxceleb2/simclr/exps/ssps_aug-none/config.yml"
    "models/ssps/voxceleb2/simclr/exps/baseline_aug-none/config.yml"

    # k-nn: M
    "models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-1/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-10/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-50/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_knn_uni-100/config.yml"

    # k-means: K, M
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_6k_uni-1/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_10k/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_10k_uni-1/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k_uni-3/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k_uni-5/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_25k_uni-10/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_50k/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_50k_uni-1/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_75k/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_75k_uni-1/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_150k/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans_150k_uni-1/config.yml"

    # Repr/Centroid
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans-centroid_25k/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_kmeans-centroid_25k_uni-1/config.yml"

    # Ref frame
    "models/ssps/voxceleb2/simclr/exps/ssps_frame-2s-clean/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_frame-3s-clean/config.yml"
    "models/ssps/voxceleb2/simclr/exps/ssps_frame-2s-aug/config.yml"

    # Margins
    "models/ssps/voxceleb2/simclr_margins/ssps_curricularface_m-0.05/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_curricularface_m-0.1/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_curricularface_m-0.2/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_adaface_m-0.05/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_adaface_m-0.1/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_adaface_m-0.2/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_arcface_m-0.05/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_arcface_m-0.1/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_arcface_m-0.2/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_cosface_m-0.05/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_cosface_m-0.1/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_cosface_m-0.2/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_magface/config.yml"
    "models/ssps/voxceleb2/simclr_margins/ssps_sphereface_m-0.1/config.yml"
)


for config in "${MODELS[@]}"; do
    torchrun --nproc_per_node=2 sslsv/bin/create_ssps_buffers_distributed.py "$config" --silent

    ./train_ddp.sh 2 "$config"

    # ./evaluate_ddp.sh 2 "$config"
    python sslsv/bin/average_model.py "$config" --silent --count 5 --limit_nb_epochs 10
    torchrun --nproc_per_node=2 sslsv/bin/evaluate_distributed.py "$config" --model_suffix avg --silent
done
