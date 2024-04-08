import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json

import torch
from torch.nn.parallel import DistributedDataParallel

from sslsv.utils.distributed import is_main_process
from sslsv.utils.helpers import load_config, load_model, evaluate as evaluate_
from evaluate import print_metrics


def evaluate(args):
    world_size = int(os.environ["WORLD_SIZE"])  # idr_torch.size
    rank = int(os.environ["LOCAL_RANK"])  # idr_torch.rank

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    config = load_config(args.config, verbose=not args.silent)

    model = load_model(config).to(rank)
    checkpoint = torch.load(config.experiment_path / "model_latest.pt")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[rank])

    metrics = evaluate_(model, config, rank, verbose=not args.silent)

    if is_main_process():
        if args.silent:
            print(metrics)
        else:
            print_metrics(metrics)

        with open(config.experiment_path / "evaluation.json", "w") as f:
            json.dump(metrics, f, indent=4)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to model config file.")
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to hide status messages and progress bars.",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Whether to save embeddings of test utterances.",
    )
    args = parser.parse_args()

    evaluate(args)
