#!/bin/bash

models="$@"

commands=""

for model in $models; do
  commands+="
python sslsv/bin/average_model.py $model/config.yml --silent# --limit_nb_epochs 80
srun python -u sslsv/bin/evaluate_distributed_jz.py $model/config.yml --model_suffix avg --silent
srun python -u sslsv/bin/inference_distributed_jz.py $model/config.yml --input 'data/voxceleb1/*/*/*.wav' --output $model/embeddings_vox1_avg.pt --model_suffix avg
"
done

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=sslsv_eval
#SBATCH --output=slurm_%j
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
###SBATCH --constraint v100-32g
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=kdp@v100

module purge

module load pytorch-gpu/py3/1.12.1

$commands
EOT