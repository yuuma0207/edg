import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, OneHotCategorical
import numpy as np
import math

# --------------------------------------------------------
# 1. Encoder / Decoder の定義
# --------------------------------------------------------
class Encoder(nn.Module):
    """
    MLPベースのEncoder: x -> (mu_z, logvar_z)
    """
    def __init__(self, x_dim=2, z_dim=2, hidden_dim=32):
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

    def forward(self, x):
        h = self.net(x)
        mu_z = self.fc_mu(h)
        logvar_z = self.fc_logvar(h)
        return mu_z, logvar_z


class Decoder(nn.Module):
    """
    MLPベースのDecoder: z -> (mu_x, logvar_x)
    """
    def __init__(self, z_dim=2, x_dim=2, hidden_dim=32):
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

    def forward(self, z):
        h = self.net(z)
        mu_x = self.fc_mu(h)
        logvar_x = self.fc_logvar(h)
        return mu_x, logvar_x


# --------------------------------------------------------
# 2. VAE モデルの定義
# --------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, x_dim=2, z_dim=2, hidden_dim=32):
        super(VAE, self).__init__()
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, x_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
          z = mu + sigma * eps
          ただし eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # エンコード
        mu_z, logvar_z = self.encoder(x) # このxはQ(x|z)のxでいい？
        # サンプリング
        z = self.reparameterize(mu_z, logvar_z)
        # デコード
        mu_x, logvar_x = self.decoder(z)
        return mu_x, logvar_x, mu_z, logvar_z



# --------------------------------------------------------
# 3. 損失関数
# --------------------------------------------------------
def loss_function(x_batch, z_batch, mu_x, logvar_x, mu_z, logvar_z, U):
    """
    VAEの損失 
    L = 1/2 * E_{Q(x|z)}[L_1(x)] + E_{Q(x|z)pi(z)}[L_2(x)]
    L_1(x) = log(σ_φ^2) + d/σ_φ^2 + |μ_φ|^2/σ_φ^2
    L_2(x) = log(Q(x|z)) + U(x)
    """
    d = z_batch.shape[1] # z.shape: (batch_size, z_dim)
    dist = Normal(mu_z, torch.exp(0.5 * logvar_z))

    # L1 = log(σ_φ^2) + d/σ_φ^2 + |μ_φ|^2/σ_φ^2
    # batch_sizeのTensorになる
    L1 = 0.5 * (logvar_x + d/torch.exp(logvar_x) + torch.norm(mu_x, dim=1)**2/torch.exp(logvar_x))

    # L2 = log(Q(x|z)) + U(x)
    # dist.log_prob(torch.tensor([1.0,2.0], [3.0,4.0]))を渡したときにどうなる？
    L2 = dist.log_prob(x_batch) + U(x_batch)

    return (L1.sum(dim=1) + L2.sum(dim=1)).mean()



# --------------------------------------------------------
# 4. 学習ループ (簡易)
# --------------------------------------------------------
def train_vae(model, data_loader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs): # 1エポックでQ(x|z)を更新する
        model.train()
        total_loss = 0.0
        batch_size = 64

        z_batch = torch.randn(batch_size, 2) # N(0, I) からサンプリング
        x_batch = model.decoder(z_batch)
        
        # 順伝播
        mu_x, logvar_x, mu_z, logvar_z = model(x_batch)
        # 損失
        loss = loss_function(x_batch, z_batch, mu_x, logvar_x, mu_z, logvar_z)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
        avg_loss = total_loss / len(data_loader)
        print(f"[Epoch {epoch+1}/{epochs}] loss: {avg_loss:.4f}")


# --------------------------------------------------------
# 5. DataLoader の例 (ダミーの 2D データ)
# --------------------------------------------------------
# ここでは乱数生成した 2D 点を使って例示
# 実際には独自のデータセットに置き換えてください
dummy_data = torch.randn(1000, 2)
dataset = torch.utils.data.TensorDataset(dummy_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# --------------------------------------------------------
# 6. 実際に VAE を作って学習
# --------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(x_dim=2, z_dim=2, hidden_dim=32).to(device)

train_vae(model, data_loader, epochs=10, lr=1e-3)

# 学習後のサンプリング
with torch.no_grad():
    # samples = model.generate(num_samples=10)
    # print("Generated samples:", samples)
    0
