import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict

import argparse
from tqdm import tqdm
from pathlib import Path
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from sslsv.bin.train import MethodWrapper
from sslsv.Config import Config
from sslsv.methods._BaseMethod import BaseMethod
from sslsv.utils.helpers import load_config, load_model, seed_dataloader_worker
from sslsv.datasets.Dataset import Dataset, DatasetConfig
from sslsv.utils.distributed import get_world_size, is_dist_initialized, is_main_process


def ddp_sync_embeddings(embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Sync embeddings across all distributed processes.

    Args:
        embeddings (Dict[str, torch.Tensor]): Dictionary containing embeddings of the local process.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing synchronized embeddings across all processes.
    """
    embeddings_all = [None for _ in range(get_world_size())]
    torch.distributed.all_gather_object(embeddings_all, embeddings)
    embeddings = {}
    for d in embeddings_all:
        embeddings.update(d)
    return embeddings


def inference_(
    config: Config,
    model: BaseMethod,
    device: torch.device,
    input: str,
    output: str,
    batch_size: int,
    frame_length: int,
    num_frames: int,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Perform model inference on a set of audio files and save the resulting embeddings.

    Args:
        config (Config): Global configuration.
        model (BaseMethod): Model used for inference.
        device (torch.device): Device on which tensors will be allocated.
        input (str): Path to input audio files with glob format (e.g. folder/*.wav).
        output (str): Output file path for saving the embeddings.
        batch_size (int): Batch size for inference.
        frame_length (int): Length of the frames to extract from the audio files.
        num_frames (int): Number of frames to extract from each audio file.
        verbose (bool): Whether to display status messages and progress bars. Defaults to True.

    Returns:
        torch.Tensor: Embeddings extracted from the model.

    Raises:
        AssertionError: If the batch size is not 1 when frame length is None.
    """
    if frame_length is None and batch_size != 1:
        raise AssertionError("Batch size must be set to 1 when frame length is None.")

    dataset_config = DatasetConfig(
        frame_length=frame_length,
        base_path=Path(""),
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )

    files = glob(input)
    dataset = Dataset(dataset_config, files, num_frames=num_frames)

    sampler = None
    if is_dist_initialized():
        sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=max(1, batch_size // get_world_size()),
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        worker_init_fn=seed_dataloader_worker,
    )

    embeddings = {}

    if is_main_process():
        dataloader = (
            tqdm(dataloader, desc="Extracting embeddings") if verbose else dataloader
        )
    for idx, X, info in dataloader:
        if X.ndim == 2:
            X = X.unsqueeze(1)

        X = X.to(device)
        B, N, L = X.size()

        X = X.reshape((B * N, L))

        with torch.no_grad():
            Y = model(X)

        Y = Y.reshape((B, N, -1))

        Y = F.normalize(Y, p=2, dim=-1)

        embeddings.update({info["files"][i]: Y[i].cpu() for i in range(B)})

    if is_dist_initialized():
        embeddings = ddp_sync_embeddings(embeddings)

    if is_main_process():
        torch.save(embeddings, output)


def inference(args: argparse.Namespace):
    """
    Perform model inference from the CLI.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.

    Returns:
        None
    """
    config = load_config(args.config, verbose=not args.silent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config).to(device)

    checkpoint = torch.load(config.model_ckpt_path / f"model_{args.model_suffix}.pt")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    if device == torch.device("cuda"):
        model = torch.nn.DataParallel(model)
    else:
        model = MethodWrapper(model)

    inference_(
        config,
        model,
        device,
        args.input,
        args.output,
        batch_size=args.batch_size,
        frame_length=args.frame_length,
        num_frames=args.num_frames,
        verbose=not args.silent,
    )


def inference_parser():
    """
    Create an argument parser for model inference.

    Returns:
        argparse.ArgumentParser: Argument parser configured with the arguments for model inference.

    Args:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to model config file.")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input audio files with glob format (e.g. folder/*.wav).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to output .pt file containing a dict of all embeddings.",
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default="latest",
        help="Model checkpoint suffix (e.g. latest, avg, ...).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--frame_length",
        type=int,
        default=None,
        help="Length of input frames.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1,
        help="Number of input frames",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Whether to hide status messages and progress bars.",
    )
    return parser


if __name__ == "__main__":
    parser = inference_parser()
    args = parser.parse_args()
    inference(args)
