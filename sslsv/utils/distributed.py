from typing import Tuple

import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    """
    Check if the distributed training environment is initialized.

    Returns:
        bool: True if the distributed training environment is initialized, False otherwise.
    """
    if dist.is_available() and dist.is_initialized():
        return True
    return False


def get_world_size() -> int:
    """
    Get the total number of processes in the distributed training environment.

    Returns:
        int: Total number of processes in the distributed training environment.
    """
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank() -> int:
    """
    Get the rank of the local process in the distributed training environment.

    Returns:
        int: Rank of the local process. Returns 0 if distributed is not initialized.
    """
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process() -> bool:
    """
    Check if the local process is the main process.

    Returns:
        bool: True if the local process is the main process, False otherwise.
    """
    return get_rank() == 0


class GatherLayer(torch.autograd.Function):
    """
    Module to gather PyTorch tensors from all distributed processes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def gather(X: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all distributed processes.

    Args:
        X (torch.Tensor): Input tensor to gather.

    Returns:
        torch.Tensor: Gathered tensor from all distributed processes.
    """
    return torch.cat(GatherLayer.apply(X)) if is_dist_initialized() else X
