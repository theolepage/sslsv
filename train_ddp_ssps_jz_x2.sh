#!/bin/bash

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=sslsv_$1
#SBATCH --output=$1/slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --constraint=a100
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@a100

module purge

module load arch/a100
module load pytorch-gpu/py3/1.12.1

# srun python -u sslsv/bin/inference_distributed_jz.py $1/config.yml --input "data/voxceleb1/*/*/*.wav" --output $1/vox2_embeddings_latest.pt --model_suffix latest

srun python -u sslsv/bin/create_ssps_buffers_distributed_jz.py $1/config.yml
srun python -u sslsv/bin/train_distributed_jz.py $1/config.yml
python sslsv/bin/average_model.py $1/config.yml --silent
srun python -u sslsv/bin/evaluate_distributed_jz.py $1/config.yml --model_suffix avg --silent
EOT
