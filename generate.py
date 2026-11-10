from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "mini_ldm"

from .model import DiffusionTransformerFlowModel
from .train_model import select_device
from .utils import generate_samples, save_mnist_grid
from .vae import VAE


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MNIST images from an ldm.pt checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("samples.png"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.sample_steps < 2:
        parser.error("samples-per-class must be positive and sample-steps must be at least 2")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vae = VAE(**saved["vae_config"]).to(device).eval().requires_grad_(False)
    vae.load_state_dict(saved["vae_state_dict"])
    model = DiffusionTransformerFlowModel(**saved["model_config"]).to(device).eval()
    model.load_state_dict(saved["model_state_dict"])
    labels = torch.arange(saved["null_label"] + 1).repeat_interleave(args.samples_per_class)
    images = generate_samples(
        model, saved["latent_shape"], labels, vae=vae,
        num_timesteps=args.sample_steps, guidance_scale=args.guidance_scale,
        null_label=saved["null_label"], use_tqdm=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_mnist_grid(images, args.output, nrow=args.samples_per_class)
    print(f"Saved {len(images)} images to {args.output}")


if __name__ == "__main__":
    main()
