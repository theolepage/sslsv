SRC_TRIAL = 'voices/sid_dev_lists_and_keys/dev-trial-keys.lst'
DST_TRIAL = 'voices2019_dev'

res = []
with open('data/' + SRC_TRIAL) as f:
    for line in f.readlines():
        a, b, target = line.split()

        sp = a.split('-')[-7][2:]
        a = f'voices/sid_dev/sp{sp}/{a}.wav'

        b = f'voices/{b}'

        line_ = '0' if target == 'imp' else '1'
        line_ += ' ' + a + ' ' + b

        res.append(line_)

with open('data/' + DST_TRIAL, 'w') as f:
    f.write('\n'.join(res))