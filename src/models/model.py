"""GAN model definitions aligned with the notebook architecture."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network used in the notebook."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 784,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator network used in the notebook."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
