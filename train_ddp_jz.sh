#!/bin/bash
#SBATCH --job-name=sslsv
#SBATCH --output=slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
###SBATCH --constraint v100-32g
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@v100

module purge

module load pytorch-gpu/py3/1.12.1

srun python -u sslsv/bin/train_distributed_jz.py $1
