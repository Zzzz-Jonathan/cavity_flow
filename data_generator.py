import torch
from torch.utils.data import DataLoader
from scipy.interpolate import griddata
import pickle
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from parameter import dataset, my_shuffle, sampling_ratio, validation_size, device, BATCH, Validation_loader
# from parameter import collate_fn

sampled_data = np.load('data/sampled_data.npz')
train_data, train_label = sampled_data['train_data_in'], sampled_data['train_label_in']
validation_data_in, validation_label_in = sampled_data['validation_data_in'], sampled_data['validation_label_in']
validation_data_out, validation_label_out = sampled_data['validation_data_out'], sampled_data['validation_label_out']

icbc = np.load('data/icbc.npz')
icbc_data, icbc_label = icbc['bc_data'], icbc['bc_label']

collocation_point = np.load('data/collocation.npy')

train_dataset = dataset(torch.FloatTensor(train_data),
                        torch.FloatTensor(train_label),
                        torch.FloatTensor(icbc_data),
                        torch.FloatTensor(icbc_label),
                        torch.FloatTensor(collocation_point))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH,
                              shuffle=True, num_workers=4, drop_last=True)
validation_dataloader = Validation_loader(torch.FloatTensor(validation_data_in).requires_grad_(False).to(device),
                                          torch.FloatTensor(validation_label_in).requires_grad_(False).to(device),
                                          torch.FloatTensor(validation_data_out).requires_grad_(False).to(device),
                                          torch.FloatTensor(validation_label_out).requires_grad_(False).to(device),
                                          batch=10 * BATCH)


def save_boundary_collocation_data(path='data/'):
    bc_data_1 = np.array([[[Re, 0, y] for Re in np.arange(0, 12.9, 0.05)]
                          for y in np.arange(0 - 1e-2, 1 + 1e-2, 0.002)])
    bc_data_2 = np.array([[[Re, 1, y] for Re in np.arange(0, 12.9, 0.05)]
                          for y in np.arange(0 - 1e-2, 1 + 1e-2, 0.002)])
    bc_data_3 = np.array([[[Re, x, 0] for Re in np.arange(0, 12.9, 0.05)]
                          for x in np.arange(0 - 1e-2, 1 + 1e-2, 0.002)])

    bc_data_1 = bc_data_1.reshape(bc_data_1.shape[0] * bc_data_1.shape[1], 3)
    bc_data_2 = bc_data_2.reshape(bc_data_2.shape[0] * bc_data_2.shape[1], 3)
    bc_data_3 = bc_data_3.reshape(bc_data_3.shape[0] * bc_data_3.shape[1], 3)

    bc_data_zero = np.concatenate([bc_data_1, bc_data_2, bc_data_3], axis=0)
    bc_label_zero = np.zeros_like(bc_data_zero)

    bc_data_slip = np.array([[[Re, x, 1] for Re in np.arange(0, 12.9, 0.05)]
                             for x in np.arange(0 - 1e-2, 1 + 1e-2, 0.002)])
    bc_data_slip = bc_data_slip.reshape(bc_data_slip.shape[0] * bc_data_slip.shape[1], 3)

    bc_label_slip = np.array([[[fx, 0, 0] for _ in np.arange(0, 12.9, 0.05)]
                              for fx in map(lambda x: (1 - (2 * x - 1) ** 18) ** 2,
                                            np.arange(0 - 1e-2, 1 + 1e-2, 0.002))])
    bc_label_slip = bc_label_slip.reshape(bc_label_slip.shape[0] * bc_label_slip.shape[1], 3)

    bc_data = np.concatenate([bc_data_zero, bc_data_slip], axis=0)
    bc_label = np.concatenate([bc_label_zero, bc_label_slip], axis=0)

    print(bc_data.shape, bc_label.shape)

    bc_data, bc_label = my_shuffle(bc_data, bc_label)

    np.savez(path + 'icbc.npz', bc_data=bc_data, bc_label=bc_label)

    # collocation points

    collocation_point = np.array([[[[Re, x, y] for Re in np.linspace(0, 12.8, 200)]
                                   for x in np.linspace(0, 1, 300)]
                                  for y in np.linspace(0, 1, 300)])

    print(collocation_point.shape)
    collocation_point = collocation_point.reshape(
        collocation_point.shape[0] * collocation_point.shape[1] * collocation_point.shape[2], 3)

    print(collocation_point.shape)

    np.save(path + 'collocation.npy', collocation_point)


def save_training_dataset(path='data/'):
    data = np.load('data/data.npy')
    label = np.load('data/label.npy')

    l, m, n = data.shape[0], data.shape[1], data.shape[2]

    train_data, train_label = data[:int(l / 2)], label[:int(l / 2)]
    validation_data, validation_label = data[int(l / 2):l], label[int(l / 2):l]

    train_data = train_data.reshape(train_data.shape[0] * m * n, 3)
    train_label = train_label.reshape(train_label.shape[0] * m * n, 3)
    validation_data = validation_data.reshape(validation_data.shape[0] * m * n, 3)
    validation_label = validation_label.reshape(validation_label.shape[0] * m * n, 3)

    train_data, train_label, validation_data, validation_label = my_shuffle(train_data,
                                                                            train_label,
                                                                            validation_data,
                                                                            validation_label)

    print("Shuffle complete !")

    train_data_in = train_data[:int(train_data.shape[0] * sampling_ratio)]
    train_label_in = train_label[:int(train_label.shape[0] * sampling_ratio)]
    validation_data_in = train_data[int(train_data.shape[0] * sampling_ratio):
                                    int(train_data.shape[0] * sampling_ratio) + validation_size]
    validation_label_in = train_label[int(train_label.shape[0] * sampling_ratio):
                                      int(train_label.shape[0] * sampling_ratio) + validation_size]
    validation_data_out = validation_data[:validation_size]
    validation_label_out = validation_label[:validation_size]

    np.savez(path + 'sampled_data.npz',
             train_data_in=train_data_in,
             train_label_in=train_label_in,
             validation_data_in=validation_data_in,
             validation_label_in=validation_label_in,
             validation_data_out=validation_data_out,
             validation_label_out=validation_label_out)


def load_data(path, load_xy=False):
    with open(path, 'rb') as file:
        data = pickle.load(file)  # shape = 7, 513, 513

        u = data[3, :, :, None]
        v = data[4, :, :, None]
        p = data[1, :, :, None]  # shape = 513, 513

        if load_xy:
            x = data[5, :, :, None]
            y = data[6, :, :, None]

            return x, y, u, v, p

    return u, v, p


def split(x, col, row):
    len1, len2 = x.shape[0], x.shape[1]
    x_splited = []

    for i in np.arange(col, len1, 2):
        temp = []
        for j in np.arange(row, len2, 2):
            temp.append(x[int(i), int(j)])
        x_splited.append(temp)

    return np.array(x_splited)


def original_data(path):
    u, v, p = load_data(path)

    p_points = split(p, 1, 1)
    p_points = p_points.reshape(p_points.shape[0] * p_points.shape[1], 1)

    u_points = split(u, 1, 0)
    u_points = u_points.reshape(u_points.shape[0] * u_points.shape[1], 1)

    v_points = split(v, 0, 1)
    v_points = v_points.reshape(v_points.shape[0] * v_points.shape[1], 1)

    return u_points[:, 0], v_points[:, 0], p_points[:, 0], \
           u[:, :, 0], v[:, :, 0], p[:, :, 0]


def merge_boundary(x, y):
    if x.shape != y.shape:
        return None

    i, j = y.shape[0], y.shape[1]
    x[:, 0] = y[:, 0]
    x[:, j - 1] = y[:, j - 1]
    x[0, :] = y[0, :]
    x[i - 1, :] = y[i - 1, :]

    return x


def save_reference(path='original_data/'):
    x, y, _, _, _ = load_data(path + 'log2Re00.0.pkl', load_xy=True)
    aim_grid = np.concatenate([x, y], axis=2)  # shape = 7, 513, 513
    p_xy = split(aim_grid, 1, 1)
    u_xy = split(aim_grid, 1, 0)
    v_xy = split(aim_grid, 0, 1)

    p_xy = p_xy.reshape(p_xy.shape[0] * p_xy.shape[1], 2)
    p_x, p_y = p_xy[:, 0], p_xy[:, 1]
    u_xy = u_xy.reshape(u_xy.shape[0] * u_xy.shape[1], 2)
    u_x, u_y = u_xy[:, 0], u_xy[:, 1]
    v_xy = v_xy.reshape(v_xy.shape[0] * v_xy.shape[1], 2)
    v_x, v_y = v_xy[:, 0], v_xy[:, 1]
    aim_x = aim_grid[:, :, 0]
    aim_y = aim_grid[:, :, 1]

    for _, _, files in os.walk(path):
        files.sort()
        data = []
        label = []

        for f in files:
            re_num = float(re.search(r'\d+\.\d+', f).group())
            u, v, p, u_itp, v_itp, p_itp = original_data(path + f)

            u = griddata((u_x, u_y), u, (aim_x, aim_y), method='cubic')
            v = griddata((v_x, v_y), v, (aim_x, aim_y), method='cubic')
            p = griddata((p_x, p_y), p, (aim_x, aim_y), method='cubic')

            u = merge_boundary(u, u_itp)[:, :, None]
            v = merge_boundary(v, v_itp)[:, :, None]
            p = merge_boundary(p, p_itp)[:, :, None]

            uvp = np.concatenate([u, v, p], axis=2)
            re_num = re_num * np.ones_like(aim_x)[:, :, None]
            xyre = np.concatenate([re_num, aim_grid], axis=2)

            data.append(xyre)
            label.append(uvp)

            print("%.4f finished." % re_num[0, 0, 0])

        np.save('data/data.npy', np.array(data))
        np.save('data/label.npy', np.array(label))

    print("Done!")


if __name__ == '__main__':
    # save_reference()
    save_training_dataset()
    # save_boundary_collocation_data()

    pass
