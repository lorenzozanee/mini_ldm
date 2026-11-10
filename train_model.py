from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "mini_ldm"

from .model import DiffusionTransformerFlowModel
from .path import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from .sample import LatentMNISTSampler, MNISTSampler
from .trainer import LatentCFGTrainer, MNISTVAETrainer, Trainer
from .vae import VAE


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the lab VAE, then a latent DiT flow model.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--vae-checkpoint", type=Path, help="Load vae.pt and skip VAE training")
    parser.add_argument("--vae-steps", type=int, default=5000)
    parser.add_argument("--flow-steps", type=int, default=10000)
    parser.add_argument("--vae-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--flow-lr", type=float, default=4e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--vae-hidden-channels", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--vae-beta", type=float, default=10.0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def train(args: argparse.Namespace) -> Path:
    positive = ["flow_steps", "vae_batch_size", "batch_size", "checkpoint_every",
                "samples_per_class", "dim", "depth", "heads"]
    if args.vae_checkpoint is None:
        positive.append("vae_steps")
    if any(getattr(args, key) < 1 for key in positive):
        raise ValueError("Training steps, batch sizes, checkpoint interval and model sizes must be positive")
    if args.warmup_steps < 0 or args.sample_steps < 2:
        raise ValueError("warmup_steps must be nonnegative and sample_steps must be at least 2")
    if args.vae_lr <= 0 or args.flow_lr <= 0 or args.vae_beta < 0:
        raise ValueError("Learning rates must be positive and vae_beta must be nonnegative")
    if args.dim % 2 or args.dim % args.heads:
        raise ValueError("dim must be even and divisible by heads")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    mnist = MNISTSampler(args.data_dir, download=args.download).to(device)
    if max(args.vae_batch_size, args.batch_size) > len(mnist.dataset):
        raise ValueError("Batch size exceeds the training dataset size")

    vae_config = dict(data_channels=1, hidden_channels=args.vae_hidden_channels, beta=args.vae_beta)
    if args.vae_checkpoint is not None:
        saved = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=True)
        vae_config = saved["vae_config"]
    vae = VAE(**vae_config).to(device)
    if 32 % vae.downsample_factor:
        raise ValueError("VAE downsampling factor must divide the 32-pixel image size")
    if args.vae_checkpoint is not None:
        vae.load_state_dict(saved["vae_state_dict"])

    latent_shape = [vae.latent_channels, 32 // vae.downsample_factor, 32 // vae.downsample_factor]
    model_config = dict(img_size=latent_shape[1], patch_size=1, c=latent_shape[0],
                        num_layers=args.depth, dim=args.dim, heads=args.heads,
                        final_dim=10, n_classes=11)
    model = DiffusionTransformerFlowModel(**model_config).to(device)
    run_dir = args.output_dir / Trainer.random_name()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), vae_config=vae_config, model_config=model_config, latent_shape=latent_shape)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Device: {device}; latent shape: {latent_shape}; run: {run_dir}", flush=True)

    if args.vae_checkpoint is None:
        print("Stage 1: VAE training", flush=True)
        MNISTVAETrainer(mnist).train(
            model=vae, num_steps=args.vae_steps, batch_size=args.vae_batch_size,
            lr=args.vae_lr, warmup_steps=args.warmup_steps,
            ckpt_every=args.checkpoint_every, output_root=run_dir, run_name="vae",
        )
    torch.save({"vae_config": vae_config, "vae_state_dict": vae.state_dict()}, run_dir / "vae.pt")

    print("Stage 2: latent flow training (VAE frozen)", flush=True)
    latent_data = LatentMNISTSampler(mnist, vae)
    path = GaussianConditionalProbabilityPath(
        p_data=latent_data, p_simple_shape=latent_shape,
        alpha=LinearAlpha(), beta=LinearBeta(),
    ).to(device)
    trainer = LatentCFGTrainer(path, sample_steps=args.sample_steps,
                               samples_per_class=args.samples_per_class)
    trainer.train(
        model=model, num_steps=args.flow_steps, batch_size=args.batch_size,
        lr=args.flow_lr, warmup_steps=args.warmup_steps,
        ckpt_every=args.checkpoint_every, output_root=run_dir, run_name="flow",
    )
    checkpoint = run_dir / "ldm.pt"
    torch.save({
        "vae_config": vae_config, "model_config": model_config,
        "latent_shape": latent_shape, "null_label": trainer.null_label,
        "vae_state_dict": vae.state_dict(), "model_state_dict": model.state_dict(),
    }, checkpoint)
    print(f"Finished. Sampling checkpoint: {checkpoint}", flush=True)
    return run_dir


if __name__ == "__main__":
    train(make_parser().parse_args())
