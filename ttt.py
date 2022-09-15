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


a1 = torch.FloatTensor((1, 2))
a2 = torch.FloatTensor((3, 4))
a3 = torch.FloatTensor((5, 6))
a4 = torch.FloatTensor((7, 8))
a = ((a1, a2), (a3, a4))
a = torch.FloatTensor(a)
print(a)
