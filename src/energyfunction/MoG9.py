import numpy as np
import matplotlib.pyplot as plt
import torch

def U_MoG9(data_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the energy function for a Mixture of 9 Gaussians (MoG9).
    """
    # Define means for 9 Gaussians in a 3x3 grid
    mu_list = [[i, j] for i in [-3, 0, 3] for j in [-3, 0, 3]]
    var = 0.3  # Variance
    inv_var = 1 / var  # Inverse variance
    x = data_batch[:, 0]
    y = data_batch[:, 1]
    u_list = []

    # Compute energy for each Gaussian
    for mu in mu_list:
        u = 0.5 * inv_var * ((x - mu[0])**2 + (y - mu[1])**2)
        u_list.append(-u)  # Negative exponent for logsumexp

    # Stack and compute log-sum-exp for stable computation
    u_list = torch.stack(u_list, dim=1)  # Shape: (batch_size, 9)
    energies = -torch.logsumexp(u_list, dim=1) + np.log(len(mu_list))  # Normalize by number of components

    return energies.unsqueeze(1)

def sampling_from_U_MoG9(n_sample: int):
    """
    Sample random points and visualize their energy for MoG9.
    """
    x_range = (-6, 6)
    y_range = (-6, 6)
    
    # Generate random points
    random_points = torch.tensor(
        [(np.random.uniform(*x_range), np.random.uniform(*y_range)) for _ in range(n_sample)],
        dtype=torch.float32
    )

    # Compute energies and plot
    energies = torch.exp(-U_MoG9(random_points))  # Convert energies to probabilities
    plt.figure(figsize=(8, 8))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.scatter(random_points[:, 0], random_points[:, 1], c=energies.squeeze(), cmap='jet')
    plt.colorbar(label='Probability')
    plt.title('Sampling from U_MoG9')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_U_MoG9():
    """
    Plot the energy function of MoG9.
    """
    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)
    
    # Create a grid of points and compute energies
    xy_tensor = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    Z = torch.exp(-U_MoG9(xy_tensor)).view(200, 200).numpy()  # Convert energy to probability

    # Plot the energy function
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    c = ax.contourf(X, Y, Z, levels=100, cmap='jet')
    plt.colorbar(c, ax=ax, label='Probability')
    ax.set_title('2D Mixture of 9 Gaussians Energy Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def main():
    # Visualize sampled points with energy
    sampling_from_U_MoG9(1000)

    # Visualize the energy function
    plot_U_MoG9()

if __name__ == '__main__':
    main()
