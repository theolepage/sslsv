import argparse
import subprocess
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np

from prepare_dataset_utils import glob, download, extract, concatenate

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

TRIALS = [
    ('voxceleb1_test_O', 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt'),
    ('voxceleb1_test_H', 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt'),
    ('voxceleb1_test_E', 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt'),
    ('voxsrc2021_val',   'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/voxsrc2021_val.txt')
]

VOX1_TRAIN_FILE = 'voxceleb1_train'
VOX2_TRAIN_FILE = 'voxceleb2_train'


def fix_vox_structure():
    subprocess.call('mkdir voxceleb1', shell=True)
    subprocess.call('mv wav/* voxceleb1', shell=True)
    subprocess.call('rm -r wav', shell=True)
    subprocess.call('mkdir voxceleb2', shell=True)
    subprocess.call('mv dev/aac/* voxceleb2', shell=True)
    subprocess.call('rm -r dev', shell=True)
    subprocess.call('rm -r vox*.zip', shell=True)


def convert_vox2_to_wav():
    files = glob('voxceleb2/*/*/*.m4a')

    for src in tqdm(files):
        dst = src.replace('.m4a', '.wav')
        cmd = 'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s'
        cmd += ' > /dev/null 2> /dev/null'
        status = subprocess.call(cmd % (src, dst), shell=True)
        if status != 0:
            raise ValueError('Conversion to wav of %s failed' % src)

        subprocess.call('rm %s' % src, shell=True)


def split_2_ssd():
    dirs = glob('dev/aac/*')
    
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


def create_vox1_train_csv():
    # Determine test speakers
    test_speakers = set()
    with open(TRIALS[0][0]) as trials:
        for line in trials.readlines():
            parts = line.rstrip().split()
            spkr_id_a = parts[1].split('/')[1]
            spkr_id_b = parts[2].split('/')[1]
            test_speakers.add(spkr_id_a)
            test_speakers.add(spkr_id_b)

    # Retrieve list of files excluding test speakers
    all_files = glob('voxceleb1/*/*/*.wav')
    all_files.sort()
    files = [f for f in all_files if f.split('/')[-3] not in test_speakers]

    df = pd.DataFrame({
        'File': files,
        'Speaker': [f.split('/')[-3] for f in files]
    })

    df.to_csv(VOX1_TRAIN_FILE, index=False)


def create_vox2_train_csv():
    files = glob('voxceleb2/*/*/*.wav')
    files.sort()

    df = pd.DataFrame({
        'File': files,
        'Speaker': [f.split('/')[-3] for f in files]
    })

    df.to_csv(VOX2_TRAIN_FILE, index=False)


def create_vox1_train_csv_gender(test_split=0.7):
    df = pd.read_csv(VOX1_TRAIN_FILE)

    # Add gender column
    vox1_meta = pd.read_csv(
        'https://www.openslr.org/resources/49/vox1_meta.csv',
        sep='\t',
        usecols=['VoxCeleb1 ID', 'Gender']
    )
    vox1_meta = vox1_meta.rename(columns={'VoxCeleb1 ID': 'Speaker'})
    df = pd.merge(df, vox1_meta, on='Speaker', how='left')

    # Add set column
    all_speakers = df['Speaker'].unique()
    train_speakers = np.random.choice(
        all_speakers,
        size=int(test_split*len(all_speakers)),
        replace=False
    )
    df['Set'] = ['train' if spk in train_speakers else 'test' for spk in df['Speaker']]

    df.to_csv('voxceleb1_train_gender.csv', index=False)


def create_vox2_train_csv_age():
    df = pd.read_csv(VOX2_TRAIN_FILE)

    df_age_train = pd.read_csv('https://raw.githubusercontent.com/nttcslab-sp/agevoxceleb/master/utt2age.train', sep=' ', names=['Key', 'Age'])
    df_age_test  = pd.read_csv('https://raw.githubusercontent.com/nttcslab-sp/agevoxceleb/master/utt2age.test', sep=' ', names=['Key', 'Age'])
    df_age_train['Set'] = 'train'
    df_age_test['Set'] = 'test'
    df_age = pd.concat((df_age_train, df_age_test))

    df['Key'] = [f[10:-4] for f in df['File']]

    df = pd.merge(df, df_age, on='Key', how='left')

    df = df.drop(columns=['Key'])
    df.dropna(inplace=True)

    conditions = [
        ((df['Age'] >   0) & (df['Age'] <=  30)),
        ((df['Age'] >  30) & (df['Age'] <=  40)),
        ((df['Age'] >  40) & (df['Age'] <=  50)),
        ((df['Age'] >  50) & (df['Age'] <=  60)),
        ((df['Age'] >  60) & (df['Age'] <= 100))
    ]
    values = [0, 1, 2, 3, 4]
    df['Age Quantized'] = np.select(conditions, values)

    df.to_csv('voxceleb2_train_age.csv', index=False)


def create_vox_trials():
    for filename, url in TRIALS:
        status = subprocess.call('wget %s -O %s' % (url, filename), shell=True)
        if status != 0:
            raise Exception('Download of %s failed' % filename)

    VOX_TRIALS = [
        'voxceleb1_test_O',
        'voxceleb1_test_E',
        'voxceleb1_test_H',
        'voxsrc2021_val'
    ]

    # Add voxceleb1 prefix to trial files
    for trial in VOX_TRIALS:
        res = []
        with open(trial) as f:
            for line in f.readlines():
                target, a, b = line.split()

                line_ = f'{target} voxceleb1/{a} voxceleb1/{b}'

                res.append(line_)

        with open(trial, 'w') as f:
            f.write('\n'.join(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    download(VOX_DOWNLOADS)
    concatenate(VOX_CONCATENATE)
    extract(VOX_EXTRACT)
    fix_vox_structure()
    convert_vox2_to_wav()

    create_vox_trials()
    create_vox1_train_csv()
    create_vox2_train_csv()

    # create_vox1_train_csv_gender()
    # create_vox2_train_csv_age()