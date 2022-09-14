import torch
from torch.utils.data import Dataset
import numpy as np


class dataset(Dataset):
    def __init__(self, train_data, train_target, icbc_data, icbc_target, c_data):
        self.train_data = train_data
        self.train_target = train_target

        self.icbc_data = icbc_data
        self.icbc_target = icbc_target

        self.c_data = c_data

        self.t_len = train_data.size(0)
        self.c_len = c_data.size(0)
        self.icbc_len = icbc_data.size(0)

        self.c_ratio = self.c_len / self.t_len
        self.icbc_ratio = self.icbc_len / self.t_len

        self.i = 0

    def __len__(self):
        return self.t_len

    def __getitem__(self, index):
        c_idx = int(index * self.c_ratio + self.i) % self.c_len
        icbc_idx = int(index * self.icbc_ratio + self.i) % self.icbc_len

        self.i += 1
        self.i = self.i % 100

        return self.train_data[index], self.train_target[index], \
               self.icbc_data[icbc_idx], self.icbc_target[icbc_idx], self.c_data[c_idx]


class Validation_loader:
    def __init__(self, *args, batch):
        self.dataset = args
        self.len = args[0].size(0)
        self.i = 0
        self.j = 0

        self.batch = batch

    def get(self):
        if self.j > 20:
            self.dataset = my_shuffle(self.dataset)
            self.j = 0

        output = []
        for d in self.dataset:
            output.append(d[self.i:self.i + self.batch])

        self.i += self.batch
        self.i = self.i % self.len

        output = tuple(output)
        return output


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def my_shuffle(*args):
    num = len(args)
    rng_state = np.random.get_state()
    for i in range(num):
        np.random.set_state(rng_state)
        np.random.shuffle(args[i])

    return args


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH = 1024
EPOCH = 10000
LOSS = torch.nn.MSELoss().to(device)

sampling_ratio = 2 ** (-4)
validation_size = 2 ** 18
