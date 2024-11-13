#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_gpus> [args ...]"
    exit 1
fi

num_gpus=$1

shift

# torchrun --nproc_per_node=$num_gpus sslsv/bin/evaluate_distributed.py "$@"

python sslsv/bin/average_model.py "$1" --silent
torchrun --nproc_per_node=$num_gpus sslsv/bin/evaluate_distributed.py "$@" --model_suffix avg
