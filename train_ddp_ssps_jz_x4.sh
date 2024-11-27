#!/bin/bash

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=sslsv_$1
#SBATCH --output=$1/slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --constraint=a100
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@a100

module purge

module load arch/a100
module load pytorch-gpu/py3/1.12.1

srun python -u sslsv/bin/create_ssps_buffers_distributed_jz.py $1/config.yml
srun python -u sslsv/bin/train_distributed_jz.py $1/config.yml
python sslsv/bin/average_model.py $1/config.yml --silent --count 5 --limit_nb_epochs 10
srun python -u sslsv/bin/evaluate_distributed_jz.py $1/config.yml --model_suffix avg --silent
EOT
