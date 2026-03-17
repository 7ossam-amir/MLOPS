"""Training loop for the GAN model."""

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import Discriminator, Generator


class GANTrainer:
    """Train the same simple GAN used in the notebook."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        lr: float = 0.0002,
        device: str = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()
        self.g_loss_history: List[float] = []
        self.d_loss_history: List[float] = []
        self.training_log: List[str] = []

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.generator.latent_dim, device=self.device)

    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch and return the last generator/discriminator losses."""
        self.generator.train()
        self.discriminator.train()

        loss_g = torch.tensor(0.0, device=self.device)
        loss_d = torch.tensor(0.0, device=self.device)

        for (real,) in data_loader:
            real = real.to(self.device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            fake = self.generator(self._sample_noise(batch_size)).detach()
            loss_d = (
                self.loss_fn(self.discriminator(real), real_labels)
                + self.loss_fn(self.discriminator(fake), fake_labels)
            )
            self.opt_D.zero_grad()
            loss_d.backward()
            self.opt_D.step()

            generated = self.generator(self._sample_noise(batch_size))
            loss_g = self.loss_fn(self.discriminator(generated), real_labels)
            self.opt_G.zero_grad()
            loss_g.backward()
            self.opt_G.step()

        return loss_g.item(), loss_d.item()

    def train(
        self,
        data_loader: DataLoader,
        epochs: int = 50,
        verbose: bool = True,
        metric_callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        """Train for the requested number of epochs."""
        for epoch in range(epochs):
            loss_g, loss_d = self.train_epoch(data_loader)
            self.g_loss_history.append(loss_g)
            self.d_loss_history.append(loss_d)
            if metric_callback is not None:
                metric_callback(epoch + 1, loss_g, loss_d)

            log_line = f"Epoch {epoch + 1}/{epochs} | D Loss: {loss_d:.3f} | G Loss: {loss_g:.3f}"
            self.training_log.append(log_line)

            if verbose:
                print(log_line)
