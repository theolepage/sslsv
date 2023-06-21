import argparse
import os

from prepare_dataset_utils import glob

NIST_SETS = [
    ('nistsre08', '/work/doc/SPEECH_DATABASE/NIST/NIST08/sp08-11/train/data/10sec/*.sph'),
    ('nistsre10', '/work/doc/SPEECH_DATABASE/NIST/SRE10_16K/data/*/*/*.sph')
]


def create_nist_train_csv():
    for name, path in NIST_SETS:
        lines = [f'0 {line}' for line in glob(path)]

        with open(f'{name}_train', 'w') as f:
            f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    create_nist_train_csv()