import torch
import torch.nn as nn

class SpacingGenerator(nn.Module):
    """
    Simple MLP that maps latent z -> positive spacings.
    """
    def __init__(self, latent_dim=8, hidden=128, depth=3):
        super().__init__()
        layers = []
        d = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z).squeeze(-1)
        # strictly positive spacings
        return torch.nn.functional.softplus(x) + 1e-6
