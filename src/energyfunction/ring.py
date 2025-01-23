import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def U_ring(data_batch: torch.Tensor) -> torch.Tensor:
    divide: float = 0.32
    x: torch.Tensor = data_batch[:,0]
    y: torch.Tensor = data_batch[:,1]
    u: torch.Tensor = (torch.sqrt(x**2 + y**2) - 2)**2 / divide
    return u.unsqueeze(1)

def sampling_from_U_ring(n_sample):
    x_range = (-6,6)
    y_range = (-6,6)
    random_points = torch.tensor([(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n_sample)])
    energies = torch.exp(-U_ring(random_points))
    print(energies)
    exit()
    plt.figure(figsize=(8, 8))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.scatter([x for x, y in random_points], [y for x, y in random_points], c=energies, cmap='jet')
    plt.colorbar()
    plt.show()

def plot_U_ring():
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    xy_tensor = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    Z = torch.exp(-U_ring(xy_tensor)).view(100, 100).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_box_aspect(1)
    c = ax.contourf(X, Y, Z, levels=100, cmap='jet')
    plt.colorbar(c, ax=ax)
    plt.show()

def main():
    #sampling_from_U_ring(10)
    plot_U_ring()

if __name__ == '__main__':
    main()