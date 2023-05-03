TRIALS = [
    'data/voxceleb1_test_O',
    'data/voxceleb1_test_E',
    'data/voxceleb1_test_H',
    'data/voxsrc2021_val'
]

for trial in TRIALS:
    res = []
    with open(trial) as f:
        for line in f.readlines():
            target, a, b = line.split()

            line_ = f'{target} voxceleb1/{a} voxceleb1/{b}'

            res.append(line_)

    with open(trial, 'w') as f:
        f.write('\n'.join(res))
