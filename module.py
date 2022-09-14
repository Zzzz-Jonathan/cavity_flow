import torch
from parameter import gradients
from parameter import LOSS as L


# re, x, y --> u, v, p


class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.nn.Sigmoid()(x)


def active_fun():
    return Swish()


def pde_loss(re, x, y, _y):
    [u, v, p] = torch.split(_y, 1, dim=1)

    uu = u * u
    uv = u * v
    vv = v * v
    Re = 2 ** re

    zeros = torch.zeros_like(u)

    u_x = gradients(u, x)
    v_y = gradients(v, y)

    uu_x = gradients(uu, x)
    uv_y = gradients(uv, y)
    uv_x = gradients(uv, x)
    vv_y = gradients(vv, y)

    u_xx = gradients(u_x, x)
    u_yy = gradients(u, y, order=2)
    v_xx = gradients(v, x, order=2)
    v_yy = gradients(v_y, y)

    p_x = gradients(p, x)
    p_y = gradients(p, y)

    p_xx = gradients(p_x, x)
    p_yy = gradients(p_y, y)

    e1 = u_x + v_y
    e2 = uu_x + uv_y + p_x - (u_xx + u_yy) / Re
    e3 = uv_x + vv_y + p_y - (v_xx + v_yy) / Re
    e4 = p_xx + p_yy + gradients(uu_x + uv_y, x) + gradients(uv_x + vv_y, y)

    return L(e1, zeros), L(e2, zeros), L(e3, zeros), L(e4, zeros)


def data_loss(_y, label):
    [u, v, p] = torch.split(_y, 1, dim=1)
    [u_l, v_l, p_l] = torch.split(label, 1, dim=1)

    return L(u, u_l), L(v, v_l), L(p, p_l)


class Yulong_NN(torch.nn.Module):
    def __init__(self):
        super(Yulong_NN, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 50),
            torch.nn.Softplus(beta=1, threshold=20),

            torch.nn.Linear(50, 50),
            torch.nn.Softplus(beta=1, threshold=20),

            torch.nn.Linear(50, 50),
            torch.nn.Softplus(beta=1, threshold=20),

            torch.nn.Linear(50, 50),
            torch.nn.Softplus(beta=1, threshold=20),

            torch.nn.Linear(50, 50),
            torch.nn.Softplus(beta=1, threshold=20),

            torch.nn.Linear(50, 50),
            torch.nn.Softplus(beta=1, threshold=20),

            torch.nn.Linear(50, 50),
            torch.nn.Softplus(beta=1, threshold=20),

            torch.nn.Linear(50, 3),
        )

    def forward(self, x, label=None):
        out = self.net(x)

        l_pde = pde_loss(x, out)

        if label is not None:
            l_data = data_loss(out, label)

            return l_pde, l_data

        return l_pde


class ResLinear(torch.nn.Module):
    def __init__(self):
        super(ResLinear, self).__init__()

        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(3, 40)
        )

        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(40, 40),
            active_fun(),
            torch.nn.Linear(40, 40),
            active_fun(),
            torch.nn.Linear(40, 40),
            active_fun(),
            torch.nn.Linear(40, 40),
            active_fun(),
        )

        self.net3 = torch.nn.Sequential(
            torch.nn.Linear(40, 40),
            active_fun(),
            torch.nn.Linear(40, 40),
            active_fun(),
            torch.nn.Linear(40, 40),
            active_fun(),
            torch.nn.Linear(40, 40),
            active_fun(),
        )

        self.net4 = torch.nn.Sequential(
            torch.nn.Linear(40, 3)
        )

        self.res1 = torch.nn.Sequential(
            torch.nn.Linear(40, 40),
            active_fun(),
        )

        self.res2 = torch.nn.Sequential(
            torch.nn.Linear(40, 40),
            active_fun(),
        )

    def forward(self, x, label=None, pde=True):
        _re, _x, _y = None, None, None
        if pde:
            _re, _x, _y = torch.split(x, 1, dim=1)
            x = torch.cat([_re, _x, _y], dim=1)

        x1 = self.net1(x)

        x2 = self.net2(x1)
        x2 += self.res1(x1)

        x3 = self.net3(x2)
        x3 += self.res2(x2)

        out = self.net4(x3)

        l_pde, l_data = None, None

        if pde:
            l_pde = pde_loss(_re, _x, _y, out)
        if label is not None:
            l_data = data_loss(out, label)

        return l_pde, l_data


if __name__ == '__main__':
    NN = ResLinear()
    aa = torch.FloatTensor([[1, 2, 3]])
    print(NN(aa, label=11))
