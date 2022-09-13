import matplotlib.pyplot as plt
from data_generator import load_data

if __name__ == '__main__':
    u, v, p = load_data('data_new/log2Re00.0.pkl')

    plt.imshow(u)
    plt.colorbar()
    plt.show()