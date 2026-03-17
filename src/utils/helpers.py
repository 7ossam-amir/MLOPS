"""Helper utilities for visualization and evaluation."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models import Discriminator, Generator

IMAGE_SIDE = 28


def _latent_dim(generator: Generator) -> int:
    return getattr(generator, "latent_dim", 64)


def _prepare_output_path(save_path: str) -> str:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def generate_samples(
    generator: Generator,
    n_samples: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """Generate digit samples from the trained generator."""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, _latent_dim(generator), device=device)
        samples = generator(z).view(n_samples, IMAGE_SIDE, IMAGE_SIDE).cpu().numpy()
        samples = (samples + 1) / 2.0
    return samples


def plot_generated_digits(samples: np.ndarray, save_path: str = "generated_digits.png") -> None:
    """Save a grid of generated digits."""
    n = len(samples)
    rows = cols = int(n ** 0.5)
    if rows * cols < n:
        rows = 4
        cols = (n + 3) // 4

    fig, axes = plt.subplots(rows, cols, figsize=(6, 6))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(samples[i], cmap="gray")
        ax.axis("off")

    plt.suptitle("Generated Digits", fontsize=11)
    plt.tight_layout()
    plt.savefig(_prepare_output_path(save_path))
    plt.close(fig)


def plot_loss_curves(
    g_loss_history: list,
    d_loss_history: list,
    save_path: str = "loss_curves.png",
) -> None:
    """Save the generator and discriminator loss curves."""
    fig = plt.figure(figsize=(8, 4))
    plt.plot(g_loss_history, label="Generator Loss")
    plt.plot(d_loss_history, label="Discriminator Loss")
    plt.axhline(y=0.693, color="gray", linestyle="--", label="Ideal balance (~0.69)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_prepare_output_path(save_path))
    plt.close(fig)


def plot_real_vs_fake(
    real_images: torch.Tensor,
    generator: Generator,
    device: str = "cpu",
    save_path: str = "real_vs_fake.png",
) -> None:
    """Save a side-by-side comparison of real and generated digits."""
    n = min(8, len(real_images))
    real_samples = real_images[:n].view(n, IMAGE_SIDE, IMAGE_SIDE).cpu().numpy()
    real_samples = (real_samples + 1) / 2.0

    generator.eval()
    with torch.no_grad():
        fake_samples = generator(
            torch.randn(n, _latent_dim(generator), device=device)
        ).view(n, IMAGE_SIDE, IMAGE_SIDE).cpu().numpy()
        fake_samples = (fake_samples + 1) / 2.0

    fig, axes = plt.subplots(2, n, figsize=(14, 4))

    for i in range(n):
        axes[0, i].imshow(real_samples[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(fake_samples[i], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Fake", fontsize=12)
    plt.suptitle("Real vs Generated Digits", fontsize=12)
    plt.tight_layout()
    plt.savefig(_prepare_output_path(save_path))
    plt.close(fig)


def evaluate_discriminator_confidence(
    discriminator: Discriminator,
    generator: Generator,
    real_images: torch.Tensor,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Return the mean discriminator score on real and generated images."""
    discriminator.eval()
    generator.eval()

    with torch.no_grad():
        real_score = discriminator(real_images[:256].to(device)).mean().item()
        fake_images = generator(torch.randn(256, _latent_dim(generator), device=device))
        fake_score = discriminator(fake_images).mean().item()

    return real_score, fake_score


def print_evaluation_report(real_score: float, fake_score: float) -> None:
    """Print the same simple confidence summary used in the notebook."""
    print("\n--- Discriminator Confidence ---")
    print(f"  Real images -> D score: {real_score:.3f}  (ideal: ~0.5)")
    print(f"  Fake images -> D score: {fake_score:.3f}  (ideal: ~0.5)")

    if abs(real_score - 0.5) < 0.15 and abs(fake_score - 0.5) < 0.15:
        print("   GAN is well-balanced!")
    elif fake_score < 0.2:
        print("  Discriminator is too strong - generator needs more training.")
    elif fake_score > 0.8:
        print("  Generator dominates - discriminator may have collapsed.")
