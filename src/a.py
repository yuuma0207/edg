import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
from typing import Tuple
from tqdm import tqdm

# MoG9エネルギー関数（簡易版）
def U_MoG9(data_batch: torch.Tensor) -> torch.Tensor:
    mu_list = [[i, j] for i in [-3, 0, 3] for j in [-3, 0, 3]]
    var = 0.3
    inv_var = 1 / var
    x = data_batch[:, 0]
    y = data_batch[:, 1]
    u_list = []
    for mu in mu_list:
        u = 0.5 * inv_var * ((x - mu[0])**2 + (y - mu[1])**2)
        u_list.append(-u)
    u_list = torch.stack(u_list, dim=1)
    energies = -torch.logsumexp(u_list, dim=1) + torch.log(torch.tensor(len(mu_list), dtype=torch.float32))
    return energies

# Encoder
class Encoder(nn.Module):
    def __init__(self, x_dim: int = 2, z_dim: int = 2, hidden_dim: int = 32):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu_x = self.fc_mu(h)
        logvar_x = self.fc_logvar(h)
        return mu_x, logvar_x

# Decoder
class Decoder(nn.Module):
    def __init__(self, z_dim: int = 2, x_dim: int = 2, hidden_dim: int = 32):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, x_dim)
        self.fc_logvar = nn.Linear(hidden_dim, x_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        mu_z = self.fc_mu(h)
        logvar_z = self.fc_logvar(h)
        return mu_z, logvar_z

# VAE本体
class VAE(nn.Module):
    def __init__(self, x_dim: int = 2, z_dim: int = 2, hidden_dim: int = 32):
        super(VAE, self).__init__()
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, x_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, logvar_z = self.decoder(z)
        x_batch = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.encoder(x_batch)
        return mu_x, logvar_x, mu_z, logvar_z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# 損失関数
def loss_function(x_batch, z_batch, mu_x, logvar_x, mu_z, logvar_z):
    L1 = torch.sum(logvar_x + (mu_x**2 + 1) / torch.exp(logvar_x))
    L2 = torch.sum((x_batch - mu_z)**2 / torch.exp(logvar_z)) + torch.sum(torch.abs(torch.exp(logvar_z))) - 2.0 * torch.sum(U_MoG9(x_batch))
    return (L1 - L2) / len(x_batch)

# 学習ループ
def train_vae(model, z_dim, epochs, batch_size, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        z_batch = torch.randn(batch_size, z_dim).to(device)
        mu_z, logvar_z = model.decoder(z_batch)
        x_batch = model.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = model.encoder(x_batch)
        loss = loss_function(x_batch, z_batch, mu_x, logvar_x, mu_z, logvar_z)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# 可視化
def visualize_results(model, z_dim, num_samples):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim)
        mu, logvar = model.decoder(z)
        samples = model.reparameterize(mu, logvar)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=5, alpha=0.5)
    ax.set_box_aspect(1)
    ax.set_title("Generated Samples")
    plt.show()

# 実行
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(x_dim=2, z_dim=10, hidden_dim=32).to(device)
    losses = train_vae(vae, z_dim=10, epochs=100, batch_size=128, lr=1e-3, device=device)
    visualize_results(vae, z_dim=10, num_samples=1000)
