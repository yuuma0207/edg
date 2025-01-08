import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, OneHotCategorical
import numpy as np

# こっちのほうが良さそうな描き方しているけど、まだ動かない

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
        mu_z, logvar_z = self.encoder(x)
        # サンプリング
        z = self.reparameterize(mu_z, logvar_z)
        # デコード
        mu_x, logvar_x = self.decoder(z)
        return mu_x, logvar_x, mu_z, logvar_z

    def generate(self, num_samples=100):
        """
        潜在空間の標準正規分布からサンプリングして x を生成
        """
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features, device=device)
        mu_x, logvar_x = self.decoder(z)
        
        # 今回は平均 mu_x をそのまま返す例
        # (本来は分散にも従いサンプリングして良い)
        return mu_x


# --------------------------------------------------------
# 3. 損失関数 (ELBO の簡易実装)
# --------------------------------------------------------
def loss_function(x, mu_x, logvar_x, mu_z, logvar_z):
    """
    VAEの損失 (Reconstruction Loss + KL Divergence)
    """
    # 再構成誤差 (平均的に計算)
    # Normal(x|mu_x, diag(exp(logvar_x))) の対数尤度を計算
    recon_loss = 0.5 * (logvar_x + (x - mu_x)**2 / torch.exp(logvar_x)).sum(dim=1)
    # z の KL
    kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)
    # バッチ平均
    return (recon_loss + kl_loss).mean()


# --------------------------------------------------------
# 4. 学習ループ (簡易)
# --------------------------------------------------------
def train_vae(model, data_loader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch in data_loader:
            x_batch = x_batch.to(next(model.parameters()).device)
            
            # 順伝播
            mu_x, logvar_x, mu_z, logvar_z = model(x_batch)
            # 損失
            loss = loss_function(x_batch, mu_x, logvar_x, mu_z, logvar_z)
            
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
    samples = model.generate(num_samples=10)
    print("Generated samples:", samples)
