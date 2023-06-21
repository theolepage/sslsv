import argparse
import subprocess
import os
import soundfile as sf
from tqdm import tqdm

from prepare_dataset_utils import glob, download, extract

AUG_DOWNLOAD = [
    ('http://www.openslr.org/resources/28/rirs_noises.zip', 'e6f48e257286e05de56413b4779d8ffb'),
    ('http://www.openslr.org/resources/17/musan.tar.gz',    '0c472d4fc0c5141eca47ad1ffeb2a7df')
]

AUG_EXTRACT = [
    'rirs_noises.zip',
    'musan.tar.gz'
]


def fix_aug_structure():
    subprocess.call('mv RIRS_NOISES/simulated_rirs .', shell=True)
    subprocess.call('rm -r RIRS_NOISES', shell=True)
    subprocess.call('rm -r rirs_noises.zip', shell=True)
    subprocess.call('rm -r musan.tar.gz', shell=True)


def split_musan(length=16000*8, stride=16000*8):
    files = glob('musan/*/*/*.wav')

    for file in tqdm(files):
        audio, fs = sf.read(file)
        
        directory = os.path.dirname(file).replace('musan/', 'musan_split/')
        os.makedirs(directory, exist_ok=True)
        
        for st in range(0, len(audio) - length, stride):
            filename = os.path.basename(file)[:-4] + ('_%05d.wav' % (st / fs))
            filename = directory + '/' + filename
            sf.write(filename, audio[st:st+length], fs)

    subprocess.call('rm -r musan', shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    download(AUG_DOWNLOAD)
    extract(AUG_EXTRACT)
    fix_aug_structure()
    split_musan()