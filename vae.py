from __future__ import annotations

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from .model import MHA, MLP


class ResidualBlock(nn.Module):
    """Pre-activation residual block with a zero-initialized output conv."""

    def __init__(
        self,
        channels: int,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: shape (b, c, h, w)
        Returns:
            x: shape (b, c, h, w)
        """
        x_skip = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x_skip + x


class AttnBlock(nn.Module):
    """Self-attention block over flattened spatial tokens."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reshape1 = Rearrange("b c h w -> b (h w) c")
        self.norm1 = nn.LayerNorm(channels)
        self.mha = MHA(channels, 1)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = MLP([channels, 2 * channels, channels], final_init=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward residual blocks.

        Args:
            x: shape (b, c, h, w)
        Returns:
            x: shape (b, c, h, w)
        """
        b, c, h, w = x.shape
        x = self.reshape1(x)

        x_skip = x
        x = self.norm1(x)
        x = self.mha(x)
        x = x + x_skip

        x_skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + x_skip

        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class EncoderBlock(nn.Module):
    """Residual and attention blocks followed by optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        downsample_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)
        self.downsample: nn.Module | None
        if downsample_channels is not None:
            self.downsample = nn.Conv2d(
                in_channels,
                downsample_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one resolution level.

        Args:
            x: shape (b, c_in, h, w)
        Returns:
            x: shape (b, c_out, h / 2, w / 2) when downsampling, else (b, c_in, h, w)
        """
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Encoder(nn.Module):
    """Convolutional image encoder with a scalar log-variance."""

    def __init__(self, in_channels: int, hidden_channels: list[int]) -> None:
        super().__init__()
        self.init_conv = nn.Conv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=3,
            padding=1,
        )

        downsample_channels = hidden_channels[1:] + [None]
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(in_c, out_c)
                for in_c, out_c in zip(hidden_channels, downsample_channels)
            ]
        )

        z_dim = hidden_channels[-1]
        self.z_mean = nn.Sequential(
            nn.GroupNorm(1, z_dim),
            nn.Conv2d(z_dim, z_dim, kernel_size=1),
        )
        self.logvar = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image to a Gaussian posterior.

        Args:
            x: shape (b, c_in, h, w)
        Returns:
            z_mean: shape (b, c_latent, h / 2**(d-1), w / 2**(d-1))
            logvar: scalar tensor
        """
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.z_mean(x), self.logvar


class DecoderBlock(nn.Module):
    """Residual and attention blocks followed by optional upsampling."""

    def __init__(
        self,
        in_channels: int,
        upsample_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)
        self.upsample: nn.Module | None
        if upsample_channels is not None:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    in_channels,
                    upsample_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode one resolution level.

        Args:
            x: shape (b, c_in, h, w)
        Returns:
            x: shape (b, c_out, 2h, 2w) when upsampling, else (b, c_in, h, w)
        """
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class Decoder(nn.Module):
    """Convolutional latent decoder with a scalar log-variance."""

    def __init__(self, out_channels: int, hidden_channels: list[int]) -> None:
        super().__init__()
        upsample_channels = hidden_channels[1:] + [None]
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(in_c, out_c)
                for in_c, out_c in zip(hidden_channels, upsample_channels)
            ]
        )

        x_dim = hidden_channels[-1]
        self.x_mean = nn.Sequential(
            nn.GroupNorm(1, x_dim),
            nn.Conv2d(x_dim, out_channels, kernel_size=1),
        )
        self.logvar = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a latent to a Gaussian image posterior.

        Args:
            z: shape (b, c_latent, h, w)
        Returns:
            x_mean: shape (b, c_out, h * 2**(d-1), w * 2**(d-1))
            logvar: scalar tensor
        """
        for block in self.blocks:
            z = block(z)
        return self.x_mean(z), self.logvar


class VAE(nn.Module):
    """Variational autoencoder for images.

    Latent spatial size is ``input_size // downsample_factor`` and latent
    channels are ``latent_channels``, matching the inputs expected by the DiT.
    """

    def __init__(
        self,
        data_channels: int,
        hidden_channels: list[int],
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        if not hidden_channels or any(c <= 0 for c in hidden_channels):
            raise ValueError("hidden_channels must be a non-empty list of positive integers")

        self.beta = beta
        self.latent_channels = hidden_channels[-1]
        self.downsample_factor = 2 ** (len(hidden_channels) - 1)

        self._encoder = Encoder(data_channels, hidden_channels)
        self._decoder = Decoder(data_channels, list(reversed(hidden_channels)))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image to posterior parameters ``(z_mean, z_logvar)``."""
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a latent to posterior parameters ``(x_mean, x_logvar)``."""
        return self._decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct an image through a reparameterized latent sample.

        Args:
            x: shape (b, c, h, w)
        Returns:
            z_mean, z_logvar, x_mean, x_logvar
        """
        z_mean, z_logvar = self.encode(x)
        z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        x_mean, x_logvar = self.decode(z)
        return z_mean, z_logvar, x_mean, x_logvar

    def compute_loss(
        self,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        x_mean: torch.Tensor,
        x_logvar: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian reconstruction loss plus a beta-weighted KL term."""
        eps = 1e-6
        kl_loss = self.beta * (
            z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1
        ).mean()
        mse_term = (x_true - x_mean).pow(2) / (torch.exp(x_logvar) + eps)
        recon_loss = (mse_term + x_logvar).mean()
        return kl_loss + recon_loss
