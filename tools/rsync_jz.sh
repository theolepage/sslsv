source_path="."
target_path="jeanzay:~/sslsv"

rsync -azh $source_path $target_path \
    --progress \
    --force \
    --delete \
    --exclude="slurm_*" \
    --exclude="data" \
    --exclude="wandb" \
    --exclude="tensorboard" \
    --exclude="*.pt" \
    --exclude="*.json" \
    --keep-dirlinks

while inotifywait -r -e modify,create,delete $source_path
do
    rsync -azh $source_path $target_path \
          --progress \
          --force \
          --delete \
          --exclude="slurm_*" \
          --exclude="data" \
          --exclude="wandb" \
          --exclude="tensorboard" \
          --exclude="*.pt" \
          --exclude="*.json" \
          --keep-dirlinks
done
