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
