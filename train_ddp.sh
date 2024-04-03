#!/bin/bash

torchrun --nproc_per_node=2 sslsv/bin/train_distributed.py $@