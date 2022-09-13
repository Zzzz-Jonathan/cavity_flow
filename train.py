import torch
from parameter import dataset, Validation_loader, device, BATCH, EPOCH
import numpy as np
from torch.utils.data import DataLoader
from module import ResLinear

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
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH, shuffle=True, num_workers=8)
validation_dataloader = Validation_loader(torch.FloatTensor(validation_data_in),
                                          torch.FloatTensor(validation_label_in),
                                          torch.FloatTensor(validation_data_out),
                                          torch.FloatTensor(validation_label_out), batch=10 * BATCH)

if __name__ == '__main__':
    NN = ResLinear()

    for epoch in range(EPOCH):
        for t_data, t_label, i_data, i_label, c_data in train_dataloader:
            t_pde_loss, t_data_loss = NN(t_data, t_label)
            i_pde_loss, i_data_loss = NN(i_data, i_label)
            c_pde_loss, _ = NN(c_data)

