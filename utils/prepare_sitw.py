import argparse
import os

SITW_TRIALS = [
    ('sitw/dev/keys/core-core.lst', 'sitw_dev_core-core'),
    ('sitw/dev/keys/core-multi.lst', 'sitw_dev_core-multi'),
    ('sitw/eval/keys/core-core.lst', 'sitw_eval_core-core'),
    ('sitw/eval/keys/core-multi.lst', 'sitw_eval_core-multi'),
]


def create_sitw_trials():
    for src, dst in SITW_TRIALS:
        res = []

        subset = src.split('/')[1]

        with open(f'sitw/{subset}/lists/enroll-core.lst') as f:
            spk_to_file = {l.split()[0]:l.strip().split()[-1] for l in f.readlines()}

        with open(src) as f:
            for line in f.readlines():
                a, b, target = line.split()

                a = f'sitw/{subset}/{spk_to_file[a]}'
                b = f'sitw/{subset}/{b}'
                target = '0' if target == 'imp' else '1'

                line_ = f'{target} {a} {b}'
                res.append(line_)

        with open(dst, 'w') as f:
            f.write('\n'.join(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to store datasets.')
    args = parser.parse_args()

    os.chdir(args.output_path)

    create_sitw_trials()