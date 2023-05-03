from glob import glob

NIST_SETS = [
    ('nistsre08', '/work/doc/SPEECH_DATABASE/NIST/NIST08/sp08-11/train/data/10sec/*.sph'),
    ('nistsre10', '/work/doc/SPEECH_DATABASE/NIST/SRE10_16K/data/*/*/*.sph')
]

for name, path in NIST_SETS:
    lines = [f'0 {line}' for line in glob(path)]

    with open(f'data/{name}_train', 'w') as f:
        f.write('\n'.join(lines))