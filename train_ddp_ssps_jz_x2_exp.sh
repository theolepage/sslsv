#!/bin/bash

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=sslsv_$1
#SBATCH --output=$1/slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
###SBATCH --constraint v100-32g
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@v100

module purge

module load pytorch-gpu/py3/1.12.1

srun python -u sslsv/bin/create_ssps_buffers_distributed_jz.py $1/config.yml
srun python -u sslsv/bin/train_distributed_jz.py $1/config.yml
python sslsv/bin/average_model.py $1/config.yml --silent --count 5 --limit_nb_epochs 10
srun python -u sslsv/bin/evaluate_distributed_jz.py $1/config.yml --model_suffix avg --silent
EOT
