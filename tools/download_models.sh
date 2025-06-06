#!/bin/bash

base_path=/lustre/fshomisc/home/rech/genoub01/uxu84ci/sslsv/./models

models=(
    # ...
)

for model in "${models[@]}"; do
    rsync -av --relative \
        jeanzay:${base_path}/${model}/{config.yml,checkpoints/model_latest.pt} .
done

rsync -av --relative models/ bititan:~/theo/sslsv/models/export