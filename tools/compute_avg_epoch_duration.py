from datetime import datetime, timedelta
from pathlib import Path
import argparse


def compute_avg_epoch_duration(args: argparse.Namespace):
    """
    Compute the average epoch duration of a model.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    total = 0
    count = 0

    slurm_files = Path(args.config).parent.glob("slurm_*")

    for slurm_file in slurm_files:
        with open(slurm_file, "r") as f:
            durations = [
                line.strip().replace('Duration: ', '')
                for line in f.readlines()
                if 'Duration' in line
            ]

            for duration in durations:
                d = datetime.strptime(duration, "%H:%M:%S")
                td = timedelta(hours=d.hour, minutes=d.minute, seconds=d.second)
                total += td.total_seconds()
                count += 1

    res = round((total / count) / 60)
    print(f"{args.config} --- Average epoch duration: {res} minutes ({count} epochs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    args = parser.parse_args()


    compute_avg_epoch_duration(args)
