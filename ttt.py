import numpy as np
from torch.utils.data import DataLoader
from parameter import dataset
import torch


def my_shuffle(*args):
    num = len(args)
    rng_state = np.random.get_state()
    for i in range(num):
        np.random.set_state(rng_state)
        np.random.shuffle(args[i])

    return args





