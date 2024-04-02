import argparse
import os
import pandas as pd
import numpy as np

from prepare_dataset_utils import glob


def create_cremad_train_csv(test_split=0.9):
    files = glob('cremad/AudioWAV/*.wav')

    LABELS = {
        'ANG': 'Anger',
        'DIS': 'Disgust',
        'FEA': 'Fear',
        'HAP': 'Joy',
        'NEU': 'Neutral',
        'SAD': 'Sad'
    }

    df = pd.DataFrame({
        'File': files,
        'Emotion': [LABELS[f.split('/')[-1].split('_')[2]] for f in files],
    })

    df.drop(df[df['Emotion'] == 'Disgust'].index, inplace=True)

    # Add set column
    train_files = np.random.choice(
        files,
        size=int(test_split*len(files)),
        replace=False
    )
    df['Set'] = [
        'train' if f in train_files else 'test'
        for f in df['File']
    ]

    df.to_csv('cremad_train.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    create_cremad_train_csv()