from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision.utils import save_image

from .sample import MNIST_MEAN, MNIST_STD
from .simulator import EulerSimulator
from .vector_field import CFGVectorFieldODE

if TYPE_CHECKING:
    from .vae import VAE

MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    return sum(p.nelement() * p.element_size() for p in model.parameters()) + sum(
        b.nelement() * b.element_size() for b in model.buffers()
    )


def record_every(num_timesteps: int, every: int) -> torch.Tensor:
    if every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [torch.arange(0, num_timesteps - 1, every), torch.tensor([num_timesteps - 1])]
    )


def save_mnist_grid(images: torch.Tensor, save_path: str | Path, nrow: int) -> None:
    if not torch.isfinite(images).all():
        raise FloatingPointError("Generated images contain non-finite values")
    images = (images.detach().cpu() * MNIST_STD + MNIST_MEAN).clamp(0, 1)
    save_image(images, str(save_path), nrow=nrow)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    latent_shape: list[int],
    labels: torch.Tensor,
    vae: VAE | None = None,
    num_timesteps: int = 100,
    guidance_scale: float = 3.0,
    null_label: int = 10,
    use_tqdm: bool = False,
) -> torch.Tensor:
    """Integrate the learned flow from noise to data, then decode latents."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    device = next(model.parameters()).device
    labels = labels.to(device=device, dtype=torch.long)
    z = torch.randn(len(labels), *latent_shape, device=device)
    ts = torch.linspace(0, 1, num_timesteps, device=device).expand(len(labels), -1)
    ode = CFGVectorFieldODE(model, null_label, guidance_scale)
    was_training = model.training
    model.eval()
    try:
        z = EulerSimulator(ode).simulate(z, ts, y=labels, use_tqdm=use_tqdm)
        if vae is not None:
            z, _ = vae.decode(z)
    finally:
        model.train(was_training)
    return z


@torch.no_grad()
def visualize_output(
    model: nn.Module,
    path,
    samples_per_class: int = 10,
    num_timesteps: int = 100,
    guidance_scales: list[float] | None = None,
    null_label: int = 10,
    save_path: str | Path = "samples.png",
    use_tqdm: bool = False,
    vae: VAE | None = None,
) -> None:
    """Save one grid per guidance scale; rows are digits 0–9 and null."""
    if guidance_scales is None:
        guidance_scales = [1.0, 3.0, 5.0]
    labels = torch.arange(null_label + 1).repeat_interleave(samples_per_class)
    save_path = Path(save_path)
    for scale in guidance_scales:
        images = generate_samples(
            model,
            path.p_simple.shape,
            labels,
            vae=vae,
            num_timesteps=num_timesteps,
            guidance_scale=scale,
            null_label=null_label,
            use_tqdm=use_tqdm,
        )
        filename = save_path.with_name(f"{save_path.stem}_cfg_{scale:g}{save_path.suffix}")
        save_mnist_grid(images, filename, nrow=samples_per_class)
