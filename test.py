import matplotlib.pyplot as plt
from data_generator import load_data

if __name__ == '__main__':
    u, v, p = load_data('original_data/log2Re12.0.pkl')

    plt.imshow(p)
    plt.colorbar()
    plt.show()