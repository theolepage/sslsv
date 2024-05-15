from typing import Dict, Iterable, Union

import torch

import time
from datetime import timedelta
from collections import defaultdict

from sslsv.utils.distributed import is_dist_initialized, get_rank


class EpochLoggerMetric:
    """
    Metric used by EpochLogger.

    Attributes:
        fmt (str): Format string for displaying the metric value.
        total (float): Total sum of values.
        count (int): Total count of values.
    """

    def __init__(self, fmt: str = "{global_avg:.6f}"):
        """
        Initialize a EpochLoggerMetric object.

        Args:
            fmt (str): Format string for displaying the metric value. Defaults to '{global_avg:.6f}'.

        Returns:
            None
        """
        self.fmt = fmt

        self.total = 0.0
        self.count = 0

    def update(self, value: float):
        """
        Update the metric value.

        Args:
            value (float): New metric value.

        Returns:
            None
        """
        self.count += 1
        self.total += value

    def synchronize(self):
        """
        Synchronize metric across distributed processes.

        Returns:
            None
        """
        if not is_dist_initialized():
            return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def global_avg(self) -> float:
        """
        Compute the global average of the metric.

        Returns:
            float: Global average of the metric.
        """
        return self.total / self.count

    def __str__(self) -> str:
        """
        Return a string representation of the metric.

        Returns:
            str: Metric string representation.
        """
        return self.fmt.format(global_avg=self.global_avg)


class EpochLogger:
    """
    Logger used during training epochs supporting metrics synchronization when DDP is enabled.

    Attributes:
        metrics (Dict[str, EpochLoggerMetric]): Dictionary of metrics.
        delimiter (str): String delimiter for formatting logger output.
    """

    def __init__(self, delimiter: str = " | "):
        """
        Initialize an EpochLogger object.

        Args:
            delimiter (str): String delimiter for formatting logger output. Defaults to ' | '.

        Returns:
            None
        """
        self.metrics = defaultdict(EpochLoggerMetric)
        self.delimiter = delimiter

    def update(self, metrics: Dict[str, Union[torch.Tensor, float, int]]):
        """
        Update metrics.

        Args:
            metrics (Dict[str, Union[torch.Tensor, float, int]]): Dictionary of new metrics.

        Returns:
            None

        Raises:
            AssertionError: If one metric value is not of type torch.Tensor, float, or int.
        """
        for k, v in metrics.items():
            assert isinstance(v, (torch.Tensor, float, int))

            if isinstance(v, torch.Tensor):
                v = v.item()
            self.metrics[k].update(v)

    def __str__(self) -> str:
        """
        Return a string containing the name and value of each metric.

        Returns:
            str: String representation of the logger.
        """
        res = []
        for name, metric in self.metrics.items():
            res.append("{}: {}".format(name, metric))
        return self.delimiter.join(res)

    def __getitem__(self, key: str) -> EpochLoggerMetric:
        """
        Get a metric.

        Args:
            key (str): Key of the metric.

        Returns:
            EpochLoggerMetric: EpochLoggerMetric object associated with the input key.
        """
        return self.metrics[key]

    def synchronize(self):
        """
        Synchronize all metrics across distributed processes.

        Returns:
            None
        """
        for metric in self.metrics.values():
            metric.synchronize()

    def log(self, iterable: Iterable, print_freq: int = 100):
        """
        Log metrics and estimated time of completion at a certain interval of iterations.

        Args:
            iterable (Iterable): Iterable object to loop over.
            print_freq (int): Frequency to print log messages. Defaults to 100.

        Yields:
            object: Next object from the iterable.
        """
        i = 0

        last_iter_end_time = time.time()
        iter_time = EpochLoggerMetric()

        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = (
            "[{0"
            + space_fmt
            + "}/{1}] "
            + self.delimiter.join(["ETA: {eta}", "{metrics}"])
        )

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
                        metrics=str(self),
                    )
                )

            i += 1
            last_iter_end_time = time.time()
