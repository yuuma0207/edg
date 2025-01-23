import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from energyfunction import MoG2, MoG9, ring5
from typing import Tuple
from tqdm import tqdm
import os
import datetime


class Encoder(nn.Module):
    """
    P_φ(z|x)
    x -> (mu_x, logvar_x)
    """
    def __init__(self, x_dim: int = 2, z_dim: int = 2, hidden_dim: int = 32):
        super(Encoder, self).__init__()
        
        # 3層 MLP (各層 32ユニット, ReLU)
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 平均ベクトル
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        # 対角共分散(ここではlogvarを出力)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu_x = self.fc_mu(h)
        logvar_x = self.fc_logvar(h)
        return mu_x, logvar_x


class Decoder(nn.Module):
    """
    Q_θ(x|z)
    z -> (mu_z, logvar_z)
    """
    def __init__(self, z_dim:int = 2, x_dim:int = 2, hidden_dim:int = 32):
        super(Decoder, self).__init__()
        
        # 3層 MLP (各層 32ユニット, ReLU)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 再構成の平均
        self.fc_mu = nn.Linear(hidden_dim, x_dim)
        # 再構成のlogvar
        self.fc_logvar = nn.Linear(hidden_dim, x_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        mu_z = self.fc_mu(h)
        logvar_z = self.fc_logvar(h)
        return mu_z, logvar_z


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick:
        z = mu + sigma * eps
        ただし eps ~ N(0, I)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# --------------------------------------------------------
# 損失関数
# mu_x: μ_φ(x), logvar_x: log(σ_φ^2(x))
# mu_z: μ_θ(z), logvar_z: log(σ_θ^2(z))
# --------------------------------------------------------
def loss_function(
        x_batch: torch.Tensor,z_batch: torch.Tensor, 
        mu_x: torch.Tensor, logvar_x: torch.Tensor, 
        mu_z: torch.Tensor, logvar_z: torch.Tensor, energy_function: torch.Tensor) -> torch.Tensor:
    """
    VAEの損失 
    L = 1/2 * (E_{Q(x|z)}[L_1(x)] - 1/2 * E_{Q(x|z)pi(z)}[L_2(x)])

    L1 = log(σ(x)^2) + (μ_x^2 + 1)/σ(x)^2

    L2 = (x - μ_z)^2)/σ_z^2 + abs(σ_z^2) - 2U(x)
    
    # x_batch : (batch_size, x_dim)
    # z_batch : (batch_size, z_dim)
    # mu_x    : (batch_size, x_dim)
    # logvar_x: (batch_size, x_dim)
    # mu_z    : (batch_size, z_dim)
    # logvar_z: (batch_size, z_dim)
    # U       : (batch_size, 1)
    """

    # L1 = log(σ(x)^2) + (μ_x^2 + 1)/σ(x)^2
    # L2 = (x - μ_z)^2)/σ_z^2 + abs(σ_z^2) - 2U(x)

    L1 = torch.sum(logvar_x + (mu_x**2 + 1)/torch.exp(logvar_x))

    L2  = torch.sum((x_batch - mu_z)**2/torch.exp(logvar_z))
    L2 += torch.sum(torch.abs(torch.exp(logvar_z)))
    L2 += -2.0*torch.sum(energy_function(x_batch))

    return (L1 - L2)/len(z_batch)


# --------------------------------------------------------
# 2. VAE モデルの定義
# --------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, x_dim:int = 2, z_dim:int = 2, hidden_dim:int = 32, energy_function: torch.Tensor = ring5.U_ring5):
        super(VAE, self).__init__()
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, x_dim, hidden_dim)
        self.energy_function = energy_function

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_z, logvar_z = self.decoder(z)
        x_batch = reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.encoder(x_batch)
        return mu_x, logvar_x, mu_z, logvar_z
    
    def get_loss(self, z_batch: torch.Tensor) -> torch.Tensor:
        mu_z, logvar_z = self.decoder(z_batch)
        x_batch = reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.encoder(x_batch)
        return loss_function(x_batch, z_batch, mu_x, logvar_x, mu_z, logvar_z, self.energy_function)




# --------------------------------------------------------
# 4. 学習ループ (簡易)
# --------------------------------------------------------
def train_vae(model: VAE, z_dim: int = 2, epochs: int =10, batch_size: int = 64, lr: float =1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    model_info = {}
    model_info["device"] = next(model.parameters()).device
    model_info["num_parameters"] = sum(p.numel() for p in model.parameters())
    model_info["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info["batch_size"] = batch_size
    model_info["learning_rate"] = lr
    model_info["epochs"] = epochs
    model_info["z_dim"] = z_dim
    model_info["hidden_dim"] = model.encoder.fc_mu.in_features
    model_info["encoder"] = model.encoder
    model_info["decoder"] = model.decoder
    model_info["optimizer"] = optimizer
    model_info["loss_function"] = model.get_loss
    model_info["energy_function"] = str(model.energy_function).split(" ")[1]

    print("Start training VAE...")
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print(f"Training...")
    for epoch in tqdm(range(epochs)): # 1エポックでQ(x|z)を更新する
        model.train()
        total_loss = 0.0
        batch_size = batch_size

        z_batch = torch.randn(batch_size, z_dim) # N(0, I) からサンプリング
        # 損失の計算
        optimizer.zero_grad()
        loss = model.get_loss(z_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
        # avg_loss = total_loss/batch_size
        avg_loss = total_loss
        losses.append(avg_loss)
    return losses, model_info

def plot_loss_and_samples(now, losses, samples, batch_size, model_info):
    fig, axes = plt.subplots(1,2,figsize=(10,5))

    plot_losses(losses, batch_size, axes[0], model_info)
    plot_samples(samples, batch_size, axes[1], model_info)
    figname = f"training_curve_and_samples_{now}.png"
    dir = "../res/"
    filename = os.path.join(dir, figname)
    plt.savefig(filename)

def plot_losses(losses, batch_size, ax: plt.Axes, model_info):
    epochs = list(range(1,len(losses)+1))
    ax.plot(epochs, losses, marker="o", linestyle="-", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_info["energy_function"]} Training: batch_size={batch_size}")

def plot_samples(samples, batch_size, ax: plt.Axes, model_info):
    ax.scatter(samples[:, 0], samples[:, 1], s=2, c="blue", alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title(f"{model_info["energy_function"]} Samples: sample_size={len(samples)}")
    
# --------------------------------------------------------
# 6. 実際に VAE を作って学習
# --------------------------------------------------------

def main(now, epochs, batch_size, num_samples, energy_function, is_sampling=False):
    now = now.strftime('%Y%m%d_%H%M%S')
    z_dim = 10
    hidden_dim = 32
    epochs=epochs
    lr=1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(x_dim=2, z_dim=z_dim, hidden_dim=hidden_dim, energy_function=energy_function).to(device)

    losses, model_info = train_vae(model=model, z_dim=z_dim, epochs=epochs, batch_size=batch_size, lr=lr)
    # losses = []; model_info = {"energy_function": energy_function}

    if is_sampling:
        # 学習後のサンプリング
        with torch.no_grad():
            z = torch.randn(num_samples, z_dim)
            mu, logvar = model.decoder(z)
            # mu = torch.zeros(num_samples, 2); logvar = torch.zeros(num_samples, 2)
            samples = reparameterize(mu, logvar)
        plot_loss_and_samples(now, losses, samples, batch_size, model_info)

if __name__ == "__main__":
    num_samples = int(1e3)
    epochs = 400
    batch_size = int(1e5)
    now = datetime.datetime.now()
    energy_function = MoG2.U_MoG2

    main(now, epochs, batch_size, num_samples, energy_function, True)