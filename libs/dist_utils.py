from typing import List

import torch.distributed as dist


def barrier():
    if dist.is_initialized():
        dist.barrier()
    else:
        pass


def broadcast(data, src):
    if dist.is_initialized():
        dist.broadcast(data, src)
    else:
        pass


def all_gather(data: List, src):
    if dist.is_initialized():
        dist.all_gather(data, src)
    else:
        data[0] = src


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)