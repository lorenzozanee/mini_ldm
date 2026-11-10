from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision import datasets, transforms

if TYPE_CHECKING:
    from .vae import VAE

MNIST_MEAN = 0.1305
MNIST_STD = 0.2891


class Sampleable(ABC):
    """Distribution which can be sampled from."""

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: the desired number of samples
        Returns:
            samples: shape (b, d)
        """
        pass


class LabeledSampleable(ABC):
    """Distribution which can be sampled from, yielding both samples and labels."""

    @abstractmethod
    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            num_samples: the desired number of samples
        Returns:
            samples: shape (b, d)
            labels: shape (b,)
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    """Sampleable wrapper around torch.randn."""

    def __init__(self, shape: list[int], std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
        self.register_buffer("dummy", torch.zeros(1))  # tracks device via self.to(...)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)


class GMM(nn.Module, LabeledSampleable):
    """Gaussian Mixture Model distribution."""

    def __init__(
        self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor
    ):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("covariances", covariances)
        self.register_buffer("weights", weights)

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_samples: the desired number of samples
        Returns:
            samples: shape (b, n)
            labels: shape (b,)
        """
        # Perform multinomial sampling on CPU to avoid device-side assert errors
        labels = torch.multinomial(
            self.weights.cpu(), num_samples=num_samples, replacement=True
        ).to(self.means.device)

        # Sample from each mode
        samples = torch.zeros(num_samples, self.means.shape[1]).to(self.means.device)
        for idx in range(len(self.means)):
            samples[labels == idx] = (
                torch.randn_like(samples[labels == idx]) * self.covariances[idx]
                + self.means[idx]
            )

        return samples, labels


class MNISTSampler(nn.Module, LabeledSampleable):
    """Sampleable wrapper for the MNIST dataset."""

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__()
        self.dataset = datasets.MNIST(
            root=str(root),
            train=True,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )
        self.register_buffer("dummy", torch.zeros(1))  # tracks device via self.to(...)

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_samples: the desired number of samples
        Returns:
            samples: shape (batch_size, c, h, w)
            labels: shape (batch_size,)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels


class LatentMNISTSampler(nn.Module, LabeledSampleable):
    """Sample the frozen VAE posterior afresh for each MNIST batch."""

    def __init__(self, mnist: MNISTSampler, vae: VAE):
        super().__init__()
        self.mnist = mnist
        self.vae = vae.eval().requires_grad_(False)

    @torch.no_grad()
    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, labels = self.mnist.sample(num_samples)
        mean, logvar = self.vae.encode(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return z, labels
