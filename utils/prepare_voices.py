import argparse
import os

SRC_VOICES_TRIAL = 'voices/sid_dev_lists_and_keys/dev-trial-keys.lst'
DST_VOICES_TRIAL = 'voices2019_dev'


def create_voices_trials():
    res = []
    with open(SRC_VOICES_TRIAL) as f:
        for line in f.readlines():
            a, b, target = line.split()

            sp = a.split('-')[-7][2:]
            a = f'voices/sid_dev/sp{sp}/{a}.wav'

            b = f'voices/{b}'

            line_ = '0' if target == 'imp' else '1'
            line_ += ' ' + a + ' ' + b

            res.append(line_)

    with open(DST_VOICES_TRIAL, 'w') as f:
        f.write('\n'.join(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    create_voices_trials()