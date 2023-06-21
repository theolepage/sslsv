import argparse
import os
import pandas as pd

from prepare_dataset_utils import glob


def create_voxlingua107_train_csv():
    files = glob('voxlingua107/**/*/*.wav')

    df = pd.DataFrame({
        'File': files,
        'Language': [f.split('/')[-2] for f in files],
        'Set': [f.split('/')[-3] for f in files]
    })

    df.to_csv('voxlingua107_train.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    create_voxlingua107_train_csv()