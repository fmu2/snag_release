import os
import random

import numpy as np
import torch


class Logger:

    def __init__(self, filepath):

        self.filepath = filepath

    def write(self, log_str):
        print(log_str)
        with open(self.filepath, 'a') as f:
            print(log_str, file=f)


class AverageMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.mean = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count

    def item(self):
        return self.mean


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def fix_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: uncomment for CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    ## NOTE: uncomment for pytorch >= 1.8
    torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)
    return rng


def iou(pred_segs, gt_segs):
    """
    Args:
        pred_segs (float tensor, (..., 2)): predicted segments.
        gt_segs (float tensor, (..., 2)): ground-truth segments.

    Returns:
        out (float tensor, (...)): intersection over union.
    """
    ps, pe = pred_segs[..., 0], pred_segs[..., 1]
    gs, ge = gt_segs[..., 0], gt_segs[..., 1]

    overlap = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp(min=0)
    union = (pe - ps) + (ge - gs) - overlap
    out = overlap / union
    return out