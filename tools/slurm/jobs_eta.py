from glob import glob
import json
import subprocess
from datetime import datetime, timedelta

NB_EPOCHS = 100

def get_slurm_jobs():
    result = subprocess.run(['squeue', '--me'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    lines = result.stdout.strip().split('\n')

    header = lines[0].split()
    job_id_index = header.index('JOBID')
    job_name_index = header.index('NAME')
    
    jobs = []
    for line in lines[1:]:
        columns = line.split()
        job_id = columns[job_id_index]
        job_name = columns[job_name_index].replace('sslsv_', '')[:-1]
        jobs.append(job_id)
    
    return jobs

etas = []

for slurm_id in get_slurm_jobs():
    slurm_paths = glob(f'models/**/slurm_{slurm_id}', recursive=True)
    if len(slurm_paths) == 0:
        continue
    slurm_path = slurm_paths[0]

    try:
        with open(slurm_path, 'r') as f:
            slurm_file = f.readlines()

        last_epoch = [
            int(line.strip().replace('Epoch ', ''))
            for line in slurm_file
            if 'Epoch' in line
        ][-1]
        
        last_duration = [
            line.strip().replace('Duration: ', '')
            for line in slurm_file
            if 'Duration' in line
        ][-1]
        
        last_duration = datetime.strptime(last_duration, "%H:%M:%S")
        last_duration = timedelta(
            hours=last_duration.hour,
            minutes=last_duration.minute,
            seconds=last_duration.second
        )
    except:
        continue

    remaining_duration = (NB_EPOCHS - last_epoch) * last_duration

    eta = datetime.now() + remaining_duration
    eta = eta.strftime("%d-%b %H:%M")

    slurm_name = slurm_path

    etas.append((slurm_name, eta))

max_name_length = max([len(eta[0]) for eta in etas]) + 1

for name, eta in etas:
    space = ' ' * (max_name_length - len(name))
    print(f'{name} {space} -> {eta}')