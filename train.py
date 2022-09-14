import torch
from parameter import device, EPOCH
from module import ResLinear
from torch.utils.tensorboard import SummaryWriter
from data_generator import train_dataloader, validation_dataloader


def module_save(nn, optimizer, it, ep, ls, name):
    state = {'model': nn.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': ep,
             'iter': it, 'loss': ls}
    torch.save(state, 'train_history/' + name)


if __name__ == '__main__':
    print(device)
    NN = ResLinear().to(device)
    opt = torch.optim.Adam(params=NN.parameters())

    writer = SummaryWriter('./train_history')
    iter = 0
    min_loss_i, min_loss_o = 1e6, 1e6

    for epoch in range(EPOCH):
        for t_data, t_label, i_data, i_label, c_data in train_dataloader:
            opt.zero_grad()
            iter += 1

            t_data, t_label, i_data, i_label, c_data = t_data.requires_grad_(True), \
                                                       t_label.requires_grad_(True), \
                                                       i_data.requires_grad_(True), \
                                                       i_label.requires_grad_(True), \
                                                       c_data.requires_grad_(True)
            vd_i, vl_i, vd_o, vl_o = validation_dataloader.get()

            t_pde_loss, t_data_loss = NN(t_data, t_label)
            i_pde_loss, i_data_loss = NN(i_data, i_label)
            i_data_loss = (i_data_loss[0], i_data_loss[1])
            c_pde_loss, _ = NN(c_data)

            loss = sum(t_pde_loss) + sum(t_data_loss) + sum(i_pde_loss) + sum(i_data_loss) + sum(c_pde_loss)

            _, vi_loss = NN(vd_i, vl_i, pde=False)
            _, vo_loss = NN(vd_o, vl_o, pde=False)

            loss.backward()
            opt.step()

            writer.add_scalars('1_loss', {'total_loss': loss,
                                          'train_loss': sum(t_pde_loss) + sum(t_data_loss),
                                          'v_in_loss': sum(vi_loss),
                                          'v_out_loss': sum(vo_loss),
                                          'icbc_loss': sum(i_pde_loss) + sum(i_data_loss),
                                          'collocation_loss': sum(c_pde_loss)}, iter)

            writer.add_scalars('pde_loss', {'total_pde': sum(t_pde_loss) + sum(i_pde_loss) + sum(c_pde_loss),
                                            't_pde': sum(t_pde_loss),
                                            'i_pde': sum(i_pde_loss),
                                            'c_pde': sum(c_pde_loss)}, iter)

            writer.add_scalars('validation_loss', {'v_in': sum(vi_loss),
                                                   'v_out': sum(vo_loss),
                                                   'vi_u': vi_loss[0], 'vi_v': vi_loss[1], 'vi_p': vi_loss[2],
                                                   'vo_u': vo_loss[0], 'vo_v': vo_loss[1], 'vo_p': vo_loss[2]},
                               iter)

            writer.add_scalars('train_loss', {'train_loss': sum(t_pde_loss) + sum(t_data_loss),
                                              'e1': t_pde_loss[0], 'e2': t_pde_loss[1],
                                              'e3': t_pde_loss[2], 'e4': t_pde_loss[3],
                                              'u': t_data_loss[0], 'v': t_data_loss[1], 'p': t_data_loss[2]},
                               iter)

            if iter % 50 == 0:
                module_save(NN, opt, iter, epoch, sum(t_data_loss).item(), name='cavity_rec')

            if min_loss_i > sum(vi_loss).item():
                min_loss_i = sum(vi_loss).item()
                module_save(NN, opt, iter, epoch, sum(vi_loss).item(), name='cavity_i')

            if min_loss_o > sum(vo_loss).item():
                min_loss_o = sum(vo_loss).item()
                module_save(NN, opt, iter, epoch, sum(vo_loss).item(),name= 'cavity_o')
