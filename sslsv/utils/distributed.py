import torch
import torch.distributed as dist


def is_dist_initialized():
    if dist.is_available() and dist.is_initialized():
        return True
    return False


def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process():
    return get_rank() == 0


class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def gather(X):
    return torch.cat(GatherLayer.apply(X)) if is_dist_initialized() else X