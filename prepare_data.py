import argparse
import subprocess
import hashlib
import os
import shutil
import glob
import soundfile as sf
from tqdm import tqdm

VOX_DOWNLOADS = [
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa', 'e395d020928bc15670b570a21695ed96'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab', 'bbfaaccefab65d82b21903e81a8a8020'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac', '017d579a2a96a077f40042ec33e51512'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad', '7bb1e9f70fddc7a678fa998ea8b3ba19'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa', 'da070494c573e5c0564b1d11c3b20577'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab', '17fe6dab2b32b48abaf1676429cdd06f'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac', '1de58e086c5edf63625af1cb6d831528'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad', '5a043eb03e15c5a918ee6a52aad477f9'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae', 'cea401b624983e2d0b2a87fb5d59aa60'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf', 'fc886d9ba90ab88e7880ee98effd6ae9'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag', 'd160ecc3f6ee3eed54d55349531cb42e'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah', '6b84a81b9af72a9d9eecbb3b1f602e65'),
    ('http://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip',   '185fdc63c3c739954633d50379a3d102')
]

VOX_CONCATENATE = [
    ('vox1_dev_wav_parta*', 'vox1_dev_wav.zip', 'ae63e55b951748cc486645f532ba230b'),
    ('vox2_dev_aac_parta*', 'vox2_dev_aac.zip', 'bbc063c46078a602ca71605645c2a402')
]

VOX_EXTRACT = [
    'vox1_dev_wav.zip',
    'vox1_test_wav.zip',
    'vox2_dev_aac.zip'
]

AUG_DOWNLOAD = [
    ('http://www.openslr.org/resources/28/rirs_noises.zip', 'e6f48e257286e05de56413b4779d8ffb'),
    ('http://www.openslr.org/resources/17/musan.tar.gz',    '0c472d4fc0c5141eca47ad1ffeb2a7df')
]

AUG_EXTRACT = [
    'rirs_noises.zip',
    'musan.tar.gz'
]

TRIALS_URL      = 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt'
TRIALS_FILENAME = 'trials'
VOX1_TRAIN_LIST = 'voxceleb1_train_list'
VOX2_TRAIN_LIST = 'voxceleb2_train_list'


def get_md5(path):
    hash_md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(entries):
    for url, md5 in entries:
        filename = url.split('/')[-1]
        status = subprocess.call('wget %s -O %s' % (url, filename), shell=True)
        if status != 0:
            raise Exception('Download of %s failed' % filename)

        if md5 != get_md5(filename):
            raise Warning('Checksum of %s failed' % filename)


def concatenate(entries):
    for src, dst, md5 in entries:
        subprocess.call('cat %s > %s' % (src, dst), shell=True)
        subprocess.call('rm %s' % (src), shell=True)

        if md5 != get_md5(dst):
            raise Warning('Checksum of %s failed' % dst)


def extract(entries):
    for filename in entries:
        if filename.endswith('.tar.gz'):
            subprocess.call('tar xf %s' % (filename), shell=True)
        elif filename.endswith('.zip'):
            subprocess.call('unzip %s' % (filename), shell=True)


def fix_vox_structure():
    subprocess.call('mkdir voxceleb1', shell=True)
    subprocess.call('mv wav/* voxceleb1', shell=True)
    subprocess.call('rm -r wav', shell=True)
    subprocess.call('mkdir voxceleb2', shell=True)
    subprocess.call('mv dev/aac/* voxceleb2', shell=True)
    subprocess.call('rm -r dev', shell=True)
    subprocess.call('rm -r vox*.zip', shell=True)


def convert_vox2_to_wav():
    files = glob.glob('voxceleb2/*/*/*.m4a')

    for src in tqdm(files):
        dst = src.replace('.m4a', '.wav')
        cmd = 'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s'
        cmd += ' > /dev/null 2> /dev/null'
        status = subprocess.call(cmd % (src, dst), shell=True)
        if status != 0:
            raise ValueError('Conversion to wav of %s failed' % src)

        subprocess.call('rm %s' % src, shell=True)


def fix_aug_structure():
    subprocess.call('mv RIRS_NOISES/simulated_rirs .', shell=True)
    subprocess.call('rm -r RIRS_NOISES', shell=True)
    subprocess.call('rm -r rirs_noises.zip', shell=True)
    subprocess.call('rm -r musan.tar.gz', shell=True)


def split_musan(length=16000*8, stride=16000*8):
    files = glob.glob('musan/*/*/*.wav')

    for file in tqdm(files):
        audio, fs = sf.read(file)
        
        directory = os.path.dirname(file).replace('musan/', 'musan_split/')
        os.makedirs(directory, exist_ok=True)
        
        for st in range(0, len(audio) - length, stride):
            filename = os.path.basename(file)[:-4] + ('_%05d.wav' % (st / fs))
            filename = directory + '/' + filename
            sf.write(filename, audio[st:st+length], fs)

    subprocess.call('rm -r musan', shell=True)


def split_2_ssd():
    dirs = glob.glob('dev/aac/*')
    
    for src in dirs:
        filename = src.split('/')[-1]
        speaker_id = int(filename[2:])
        if speaker_id % 2 == 0:
            dst = '/diskssd1/ing2/datasets/VoxCeleb2/dev/aac/' + filename
        else:
            dst = '/diskssd2/ing2/datasets/VoxCeleb2/dev/aac/' + filename
        
            # Create symlink
            symlink_src = '/diskssd2/ing2/datasets/VoxCeleb2/dev/aac/' + filename
            symlink_dst = '/diskssd1/ing2/datasets/VoxCeleb2/dev/aac/' + filename
            os.symlink(symlink_src, symlink_dst)
        
        shutil.copytree(src, dst)


def create_vox1_train_list_file():
    test_speakers = set()
    with open(TRIALS_FILENAME) as trials:
        for line in trials.readlines():
            parts = line.rstrip().split()
            spkr_id_a = parts[1].split('/')[0]
            spkr_id_b = parts[2].split('/')[0]
            test_speakers.add(spkr_id_a)
            test_speakers.add(spkr_id_b)

    files = glob.glob('voxceleb1/*/*/*.wav')
    files.sort()
    out_file = open(VOX1_TRAIN_LIST, 'w')
    for file in files:
        spkr_id = file.split('/')[-3]
        file = '/'.join(file.split('/')[-3:])
        file = os.path.join('voxceleb1', file)
        if spkr_id not in test_speakers:
            out_file.write(spkr_id + ' ' + file + '\n')
    out_file.close()


def create_vox2_train_list_file():
    files = glob.glob('voxceleb2/*/*/*.wav')
    files.sort()
    out_file = open(VOX2_TRAIN_LIST, 'w')
    for file in files:
        spkr_id = file.split('/')[-3]
        file = '/'.join(file.split('/')[-3:])
        file = os.path.join('voxceleb2', file)
        out_file.write(spkr_id + ' ' + file + '\n')
    out_file.close()


def download_trials_file():
    status = subprocess.call('wget %s -O %s' % (TRIALS_URL, TRIALS_FILENAME), shell=True)
    if status != 0:
        raise Exception('Download of %s failed' % TRIALS_FILENAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    # VoxCeleb1 and VoxCeleb2
    download(VOX_DOWNLOADS)
    concatenate(VOX_CONCATENATE)
    extract(VOX_EXTRACT)
    fix_vox_structure()
    convert_vox2_to_wav()

    # Augmentation: MUSAN and simulated_rirs
    download(AUG_DOWNLOAD)
    extract(AUG_EXTRACT)
    fix_aug_structure()
    split_musan()

    download_trials_file()
    create_vox1_train_list_file()
    create_vox2_train_list_file()