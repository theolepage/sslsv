import torch

import time
from datetime import timedelta
from collections import defaultdict

from sslsv.utils.distributed import (
    is_dist_initialized,
    get_rank
)


class EpochLoggerMetric(object):

    def __init__(self, fmt='{global_avg:.6f}'):
        self.fmt = fmt

        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.count += 1
        self.total += value

    def synchronize(self):
        if not is_dist_initialized(): return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def global_avg(self):
        return self.total / self.count
    
    def __str__(self):
        return self.fmt.format(global_avg=self.global_avg)


class EpochLogger:

    def __init__(self, delimiter=' | '):
        self.metrics = defaultdict(EpochLoggerMetric)
        self.delimiter = delimiter

    def update(self, metrics):
        for k, v in metrics.items():
            assert isinstance(v, (torch.Tensor, float, int))

            if isinstance(v, torch.Tensor): v = v.item()
            self.metrics[k].update(v)

    def __str__(self):
        res = []
        for name, metric in self.metrics.items():
            res.append('{}: {}'.format(name, metric))
        return self.delimiter.join(res)

    def __getitem__(self, key):
        return self.metrics[key]

    def synchronize(self):
        for metric in self.metrics.values():
            metric.synchronize()

    def log(self, iterable, print_freq=100):
        i = 0

        last_iter_end_time = time.time()
        iter_time = EpochLoggerMetric()
        
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = '[{0' + space_fmt + '}/{1}] ' + self.delimiter.join([
            'ETA: {eta}',
            '{metrics}'
        ])

        for obj in iterable:
            yield obj
            iter_time.update(time.time() - last_iter_end_time)
            if get_rank() == 0 and (i % print_freq == 0 or i == len(iterable) - 1):
                eta = int(iter_time.global_avg * (len(iterable) - i))
                print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=str(timedelta(seconds=eta)),
                        metrics=str(self)
                    )
                )

            i += 1
            last_iter_end_time = time.time()