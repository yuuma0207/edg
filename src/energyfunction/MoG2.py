import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def U_MoG2(data_batch: torch.Tensor) -> torch.Tensor:
    mu1 = [-5.0,0]
    mu2 = [5.0,0]
    var: float = 0.5
    x: torch.Tensor = data_batch[:,0]
    y: torch.Tensor = data_batch[:,1]
    u_1 = 0.5 * ((x - mu1[0])**2 + (y - mu1[1])**2) / var
    u_2 = 0.5 * ((x - mu2[0])**2 + (y - mu2[1])**2) / var
    # logsumexp([-u1, -u2]): log(exp(-u_1) + exp(-u_2))
    energies = -torch.logsumexp(-torch.stack([u_1, u_2], dim=1), dim=1)
    return -energies.unsqueeze(1)

def sampling_from_U_MoG2(n_sample):
    x_range = (-6,6)
    y_range = (-6,6)
    random_points = torch.tensor([(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n_sample)])
    energies = torch.exp(-U_MoG2(random_points))
    plt.figure(figsize=(8, 8))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.scatter([x for x, y in random_points], [y for x, y in random_points], c=energies, cmap='jet')
    plt.colorbar()
    plt.show()

def plot_U_MoG2():
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    xy_tensor = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    Z = torch.exp(-U_MoG2(xy_tensor)).view(100, 100).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_box_aspect(1)
    c = ax.contourf(X, Y, Z, levels=100, cmap='jet')
    plt.colorbar(c, ax=ax)
    plt.show()

def main():
    # sampling_from_U_MoG2(1000)
    plot_U_MoG2()

if __name__ == '__main__':
    main()