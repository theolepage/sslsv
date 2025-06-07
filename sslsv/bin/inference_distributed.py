import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse

import torch
from torch.nn.parallel import DistributedDataParallel

from sslsv.bin.inference import inference_parser, inference_
from sslsv.utils.helpers import load_config, load_model


def inference(args: argparse.Namespace):
    """
    Perform model inference from the CLI (using DistributedDataParallel).

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    world_size = int(os.environ["WORLD_SIZE"])  # idr_torch.size
    rank = int(os.environ["LOCAL_RANK"])  # idr_torch.rank

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    config = load_config(args.config, verbose=not args.silent)

    model = load_model(config).to(rank)
    checkpoint = torch.load(config.model_ckpt_path / f"model_{args.model_suffix}.pt")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[rank])

    inference_(
        config,
        model,
        torch.device("cuda", rank),
        args.input,
        args.output,
        batch_size=args.batch_size,
        frame_length=args.frame_length,
        num_frames=args.num_frames,
        verbose=not args.silent,
    )


if __name__ == "__main__":
    parser = inference_parser()
    args = parser.parse_args()
    inference(args)
