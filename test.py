import matplotlib.pyplot as plt
from data_generator import load_data
import numpy as np
from module import ResLinear, module_load
import torch
from parameter import gradients

path = 'train_history/sgm/cavity_rec'
path_reference = 'original_data/log2Re01.0.pkl'
RE = 1


def my_imshow(out, l=513, m=513, usable=False, vmin=None, vmax=None):
    if not usable:
        out = out.reshape(l, m, out.shape[-1])

    U = []
    for i in range(out.shape[-1]):
        U.append(out[:, :, i])

    for idx, i in enumerate(U):
        plt.imshow(i, vmin=vmin[idx] if vmin is not None else None,
                   vmax=vmax[idx] if vmin is not None else None,
                   cmap=plt.get_cmap('seismic'))
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    NN = ResLinear()

    x, y, u, v, p = load_data(path_reference, load_xy=True)
    omega_reference = load_data(path_reference, omega=True)
    uvp_reference = np.concatenate([u, v, p], axis=2)
    vmin, vmax = np.min(uvp_reference, axis=(0, 1)), np.max(uvp_reference, axis=(0, 1))
    re = np.ones_like(x) * RE
    rexy = np.concatenate([re, x, y], axis=2).reshape(re.shape[0] * re.shape[1], 3)

    nn_para, _, _, _, _ = module_load(path)
    NN.load_state_dict(nn_para)

    nn_out = NN(torch.FloatTensor(rexy), test=True).detach().numpy()
    omega = NN(torch.FloatTensor(rexy).requires_grad_(True), test_omega=True).detach().numpy()

    # my_imshow(nn_out, vmin=vmin, vmax=vmax)

    my_imshow(omega)
    my_imshow(omega_reference, usable=True)
    my_imshow(nn_out)
    my_imshow(uvp_reference, usable=True)