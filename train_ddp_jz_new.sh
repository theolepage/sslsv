#!/bin/bash

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=sslsv___$1
#SBATCH --output=$1slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
###SBATCH --constraint v100-32g
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@v100

module purge

module load pytorch-gpu/py3/1.12.1

srun python -u slurm_train_distributed.py $1config.yml
EOT
