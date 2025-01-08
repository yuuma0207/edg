import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Define the score function model
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ScoreNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, z, x, t):
        t = t.expand(-1,1)
        return self.net(torch.cat([z, x, t], dim=-1))

# Define the Hamiltonian-based decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.qv = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.tv = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, y, v, energy_grad):
        step_size = 0.01  # Step size for Hamiltonian updates
        v = v - 0.5 * step_size * energy_grad
        y = y + step_size * v
        v = v - 0.5 * step_size * energy_grad
        return y, v

# Define the energy function (example: quadratic energy)
def energy_function(x):
    return 0.5 * torch.sum(x ** 2, dim=-1)

# Training loop
input_dim = 10
hidden_dim = 64

decoder = Decoder(input_dim, hidden_dim)
score_net = ScoreNetwork(input_dim, hidden_dim)
optimizer = optim.Adam(list(decoder.parameters()) + list(score_net.parameters()), lr=1e-3)

def train_step(batch_x):
    batch_size = batch_x.size(0)
    batch_x.requires_grad = True
    z = torch.randn(batch_size, input_dim)
    t = torch.rand(batch_size, 1)  # Time variable


    # Forward pass
    energy = energy_function(batch_x)
    energy_grad = torch.autograd.grad(energy_function(batch_x).sum(), batch_x, create_graph=True)[0]
    y, v = decoder(z, z, energy_grad)
    score = score_net(y, batch_x, t)

    # Loss computation (KL divergence upper bound approximation)
    loss = torch.mean((score - energy_grad) ** 2)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Example training loop
epochs = 10
for epoch in range(epochs):
    batch_x = torch.randn(32, input_dim, requires_grad=True)  # Example batch
    loss = train_step(batch_x)
    print(f"Epoch {epoch + 1}, Loss: {loss}")
