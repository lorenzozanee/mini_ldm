# Latent Flow Matching on MNIST

`mini_ldm` is a minimal **latent diffusion** implementation: the lab VAE compresses `1×32×32` MNIST images to `128×4×4` latents, and a Diffusion Transformer (DiT) learns a conditional flow-matching vector field on that latent space with Classifier-Free Guidance (CFG).

Based on the formulation in *Flow Matching for Generative Modeling* (Lipman et al., 2023).

## Environment Setup

```bash
# from the repository parent directory (the one containing mini_ldm/)
conda env create -f mini_ldm/environment.yaml
conda activate mini_ldm
```

## Quick Start

```bash
# from the repository parent directory
python -m mini_ldm.train_model
```

The script trains the VAE, freezes it, trains the latent DiT, and saves a single sampling checkpoint:

```
mini_ldm/runs/<run-name>/
├── config.json
├── vae.pt                  # trained VAE
├── vae/                    # loss log, reconstructions, step checkpoints
├── flow/                   # loss log, decoded samples, step checkpoints
└── ldm.pt                  # VAE + DiT weights and configs
```

Generate images:

```bash
python -m mini_ldm.generate mini_ldm/runs/<run-name>/ldm.pt --output samples.png
```

Fewer steps are fine for a sanity check:

```bash
python -m mini_ldm.train_model --vae-steps 1000 --flow-steps 2000 \
    --vae-batch-size 16 --batch-size 32 --checkpoint-every 1000
```

`--device auto` uses CUDA, then MPS, then CPU. MNIST is read from `../data` (or `--data-dir`); use `--no-download` when it is already present.

## Structure

```
mini_ldm/
├── train_model.py       # Two-stage training entry point
├── generate.py          # Sample from ldm.pt
├── vae.py               # Lab VAE: encoder, decoder, reparameterization, loss
├── model.py             # DiT backbone (patchify → transformer → depatchify)
├── path.py              # Probability path: α_t, β_t, p_t(x|z) = N(α_t·z, β_t²·I)
├── sample.py            # MNIST sampler, frozen-VAE latent sampler
├── vector_field.py      # Conditional vector field u_t^θ(x|y) + CFG wrapper
├── simulator.py         # Euler ODE solver
├── trainer.py           # Shared loop, VAE trainer, latent CFG trainer
└── utils.py             # Sampling, grids, model size
```

## Two-Stage Training

**Stage 1 — VAE.** Same architecture and objective as the lab notebook: residual blocks with `GroupNorm(1, C)`, single-head attention, three down/up-sampling levels, a scalar encoder/decoder log-variance, `beta=10`, and the Gaussian NLL + KL loss. The default `[16, 32, 64, 128]` channels map `1×32×32 → 128×4×4 → 1×32×32`.

**Stage 2 — latent flow.** The VAE is set to `eval()` with `requires_grad_(False)`; every batch is encoded under `no_grad`, the posterior is reparameterized, and the DiT is trained on those latents. The CFG objective independently replaces labels with `null_label=10` with probability `eta=0.35`. Sampling starts from `z0 ~ N(0, I)`, integrates the CFG-guided ODE with Euler steps, and decodes the final latent with the frozen VAE.

| Concept | Code Location | Formula |
|---------|--------------|---------|
| VAE posterior | `vae.py` | `z = z_mean + exp(0.5·logvar)·ε` |
| Probability path | `path.py` | `p_t(x\|z) = N(α_t·z, β_t²·I)` |
| CFM loss | `trainer.py` | `‖u_t^θ(x\|y) - u_t(x\|z)‖²` |
| CFG inference | `vector_field.py` | `v = (1-w)·v(x,t,∅) + w·v(x,t,y)` |
| ODE sampling | `simulator.py` | `x_{t+h} = x_t + v(x_t,t)·h` |

## Defaults

| Setting | VAE | Latent DiT |
|---------|-----|------------|
| Steps | 5,000 | 10,000 |
| Batch size | 64 | 256 |
| Learning rate | 1e-3 | 4e-4 |
| Warmup | 500 | 500 |
| Other | `beta=10`, channels `[16,32,64,128]` | 8 layers, dim 256, 8 heads, patch 1 |

A reduced-step run was verified end to end on MPS; the saved `ldm.pt` also reloads on CPU with `weights_only=True`.

## Usage as a Library

```python
import torch
from mini_ldm import (
    VAE, DiffusionTransformerFlowModel, GaussianConditionalProbabilityPath,
    LinearAlpha, LinearBeta, LatentMNISTSampler, MNISTSampler,
)

device = torch.device("mps")
mnist = MNISTSampler("data", download=True).to(device)
vae = VAE(data_channels=1, hidden_channels=[16, 32, 64, 128], beta=10.0).to(device)
latents = LatentMNISTSampler(mnist, vae)
path = GaussianConditionalProbabilityPath(
    p_data=latents,
    p_simple_shape=[vae.latent_channels, 4, 4],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
)
model = DiffusionTransformerFlowModel(
    img_size=4, patch_size=1, c=vae.latent_channels,
    num_layers=8, dim=256, heads=8, final_dim=10, n_classes=11,
).to(device)
```
