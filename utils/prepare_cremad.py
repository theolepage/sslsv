import argparse
import os
import pandas as pd
import numpy as np

from prepare_dataset_utils import glob


def create_cremad_train_csv(test_split=0.7):
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

    # Add set column
    all_speakers = list(set([f.split('/')[-1].split('_')[0] for f in files]))
    train_speakers = np.random.choice(
        all_speakers,
        size=int(test_split*len(all_speakers)),
        replace=False
    )
    df['Set'] = [
        'train'
        if f.split('/')[-1].split('_')[0] in train_speakers else 'test'
        for f in df['File']
    ]

    df.to_csv('cremad_train.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    create_cremad_train_csv()