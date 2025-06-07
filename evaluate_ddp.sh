#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_gpus> [args ...]"
    exit 1
fi

num_gpus=$1

shift

# torchrun --nproc_per_node=$num_gpus sslsv/bin/evaluate_distributed.py "$@"

# python sslsv/bin/average_model.py "$1" --silent
torchrun --nproc_per_node=$num_gpus sslsv/bin/evaluate_distributed.py "$@" --model_suffix avg


# torchrun --nproc_per_node=$num_gpus sslsv/bin/inference_distributed.py "$1"/config.yml --input 'data/voxceleb1/*/*/*.wav' --output "$1"/embeddings_vox1_avg.pt --model_suffix avg
