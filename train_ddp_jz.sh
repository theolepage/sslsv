#!/bin/bash

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=sslsv_$2
#SBATCH --output=$2/slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=$1
#SBATCH --gres=gpu:$1
#SBATCH --cpus-per-task=10
####SBATCH --constraint v100-32g
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@v100

module purge

module load pytorch-gpu/py3/1.12.1

srun python -u sslsv/bin/train_distributed_jz.py $2/config.yml
python sslsv/bin/average_model.py $2/config.yml --silent
srun python -u sslsv/bin/evaluate_distributed_jz.py $2/config.yml --model_suffix avg --silent
EOT
