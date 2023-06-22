#!/bin/bash

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train_distributed.py $1