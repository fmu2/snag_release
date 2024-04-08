import os
import random

import numpy as np
import torch


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id, num_workers, rank):
    """
        Reset random seed for each worker.
    """
    seed = torch.initial_seed() % 2 ** 31   # base_seed + worker_id
    worker_seed = num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)