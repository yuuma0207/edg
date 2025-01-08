import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# デバイスの設定（GPU が使える場合は GPU、そうでなければ CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 前処理 (tensor 化 & 画素値を [0,1] に)
transform = transforms.Compose([
    transforms.ToTensor()
])

# 訓練データとテストデータの取得
train_dataset = datasets.MNIST(root="./../data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./../data", train=False,  download=True, transform=transform)

# DataLoader の作成
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)



class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor):
        # x: (batch_size, input_dim)
        # 3層の MLP
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        # 平均 (mu) と 対数分散 (logvar)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor):
        # z: (batch_size, latent_dim)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        # 出力層
        x_hat = self.fc_out(h)
        # MNIST は [0,1] の画素値なのでシグモイドをかけることが多い
        x_hat = torch.sigmoid(x_hat)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 32, latent_dim: int = 2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def encode(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z: torch.Tensor):
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

def loss_function(x_hat, x, mu, logvar):
    """
    x_hat: VAE が再構成した画像 (batch_size, input_dim)
    x    : 元の画像 (batch_size, input_dim)
    mu, logvar: Encoder 出力

    戻り値: VAE の損失 (BCE + KL)
    """
    # バイナリクロスエントロピー (ピクセルごとに計算し合計)
    bce = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # KL ダイバージェンス
    # D_KL( q(z|x) || p(z) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (bce + kld) / len(x)
    # 以下は本に書いていたやつ（こっちの方が精度出てほしいけど...）
    L1 = F.mse_loss(x_hat, x, reduction='sum')
    L2 = - torch.sum(1 + torch.log(logvar.exp() ** 2) - mu ** 2 - logvar.exp() ** 2)
    return (L1 + L2) / len(x)

# input_dim = 784
# hidden_dim = 200
# latent_dim = 20
input_dim = 784
hidden_dim = 32
latent_dim = 2

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.0

    for images, _ in train_loader:
        # (batch_size, 1, 28, 28) → (batch_size, 784)
        images = images.view(-1, 784).to(device)
        
        # 順伝播
        x_hat, mu, logvar = model(images)
        
        # 損失計算
        loss = loss_function(x_hat, images, mu, logvar)
        
        # 勾配リセット
        optimizer.zero_grad()
        
        # 逆伝播
        loss.backward()
        
        # パラメータ更新
        optimizer.step()
        
        train_loss += loss.item()
    
    # 1エポックあたりの平均損失
    avg_train_loss = train_loss / len(train_loader.dataset)

    # テスト評価
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(-1, 784).to(device)
            x_hat, mu, logvar = model(images)
            loss = loss_function(x_hat, images, mu, logvar)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader.dataset)

    print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

model.eval()
with torch.no_grad():
    # テストデータの先頭バッチを取り出し
    images, _ = next(iter(test_loader))
    images = images.to(device)
    # 元の画像と再構成画像
    x = images.view(-1, 784)
    x_hat, _, _ = model(x)
    
    # 表示用に CPU へ
    x = x.cpu().numpy()
    x_hat = x_hat.cpu().numpy()

# 先頭 8 枚を表示
n_show = 8
plt.figure(figsize=(16, 4))
for i in range(n_show):
    # 元画像
    ax = plt.subplot(2, n_show, i + 1)
    plt.imshow(x[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    
    # 再構成画像
    ax = plt.subplot(2, n_show, i + 1 + n_show)
    plt.imshow(x_hat[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle("Original (top) vs Reconstructed (bottom)", fontsize=16)
plt.show()


model.eval()
with torch.no_grad():
    # 標準正規分布からサンプリング
    z = torch.randn(batch_size, latent_dim).to(device)
    samples = model.decode(z).cpu().numpy()

# 先頭 8 枚を表示
plt.figure(figsize=(16, 2))
for i in range(n_show):
    ax = plt.subplot(1, n_show, i + 1)
    plt.imshow(samples[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle("Randomly Sampled Digits from Latent Space", fontsize=16)
plt.show()
