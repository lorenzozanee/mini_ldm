from __future__ import annotations

import math

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from .vector_field import ConditionalVectorField

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation."""

    def __init__(
        self,
        dims: list[int],
        activation: type[nn.Module] = nn.SiLU,
        final_init: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

        if final_init:
            nn.init.zeros_(self.net[-1].weight)  # type: ignore[arg-type]
            nn.init.zeros_(self.net[-1].bias)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: shape (b, n, d)
        Returns:
            x: shape (b, n, d)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Diffusion Transformer (DiT) components
# ---------------------------------------------------------------------------


class FourierEncoder(nn.Module):
    """Fourier feature time encoder.

    Based on: https://github.com/lucidrains/denoising-diffusion-pytorch
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode scalar time to Fourier features.

        Args:
            t: shape (b,)
        Returns:
            embeddings: shape (b, d)
        """
        t = t.view(-1, 1)  # (b, 1)
        freqs = t * self.weights * 2 * math.pi  # (b, half_dim)
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)


class Patchifier(nn.Module):
    """Convert image to sequence of patch tokens via strided convolution."""

    def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"Image size {img_size} must be divisible by patch size {patch_size}"
            )
        self.net = nn.Sequential(
            nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b d h w -> b (h w) d"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify image.

        Args:
            x: shape (b, c, h, w)
        Returns:
            tokens: shape (b, n_tokens, dim)
        """
        return self.net(x)


class MHA(nn.Module):
    """Multi-headed self-attention with fused QKV projection."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.qkv = nn.Linear(dim, dim * 3)
        self.fold_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.unfold_heads = Rearrange("b h n d -> b n (h d)")
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention forward pass.

        Args:
            x: shape (b, n, d)
        Returns:
            x: shape (b, n, d)
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each (b, n, head_dim*h)
        q, k, v = map(self.fold_heads, (q, k, v))  # each (b*h, n, head_dim)

        x = F.scaled_dot_product_attention(q, k, v)

        x = self.unfold_heads(x)  # (b, n, d)
        return self.out(x)


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Adaptive modulation (adaLN-Zero style): x * (1 + scale) + bias.

    Args:
        x: shape (b, n, d)
        scale: shape (b, n, d)
        bias: shape (b, n, d)
    Returns:
        x: shape (b, n, d)
    """
    return x * (1 + scale) + bias


class DiffusionTransformerLayer(nn.Module):
    """Single DiT block with adaLN-Zero conditioning."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()

        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6),
        )

        # Zero-initialize conditioning to stabilize residual at start
        nn.init.zeros_(self.ada_ln[1].weight)  # type: ignore[index]
        nn.init.zeros_(self.ada_ln[1].bias)  # type: ignore[index]

        self.attn = MHA(dim, heads)
        self.ff = MLP([dim, 4 * dim, dim])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """DiT block forward with adaLN conditioning.

        Args:
            x: shape (b, n, d) — token sequence
            c: shape (b, d) — conditioning vector (time + label embedding)
        Returns:
            x: shape (b, n, d)
        """
        # Compute modulation parameters: 6 × d split into scale/bias/gate pairs
        c = rearrange(self.ada_ln(c), "b d -> b 1 d")
        (
            attn_scale,
            attn_bias,
            attn_gate,
            ff_scale,
            ff_bias,
            ff_gate,
        ) = c.chunk(6, dim=-1)

        # Self-attention with adaLN
        x = x + attn_gate * self.attn(modulate(self.norm1(x), attn_scale, attn_bias))
        # Feed-forward with adaLN
        x = x + ff_gate * self.ff(modulate(self.norm2(x), ff_scale, ff_bias))
        return x


class DiffusionTransformer(nn.Module):
    """Stack of DiT blocks with learned positional encodings."""

    def __init__(
        self,
        depth: int,
        n_tokens: int,
        dim: int,
        **layer_kwargs: object,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DiffusionTransformerLayer(dim=dim, **layer_kwargs) for _ in range(depth)]
        )
        self.pos_encodings = nn.Parameter(torch.randn(n_tokens, dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass through all DiT blocks.

        Args:
            x: shape (b, n, d) — patch tokens
            c: shape (b, d) — conditioning
        Returns:
            x: shape (b, n, d)
        """
        x = x + self.pos_encodings.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, c)
        return x


class Depatchifier(nn.Module):
    """Convert token sequence back to image via MLP + rearrange + conv."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        dim: int,
        final_dim: int,
        c_out: int,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"Image size {img_size} must be divisible by patch size {patch_size}"
            )
        self.patch_size = patch_size
        h = w = img_size // patch_size

        self.net = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            MLP([dim, 4 * dim, final_dim * patch_size**2]),
            Rearrange(
                "b (h w) (f ph pw) -> b f (h ph) (w pw)",
                h=h,
                w=w,
                f=final_dim,
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Depatchify tokens back to image.

        Args:
            x: shape (b, n, d)
        Returns:
            img: shape (b, c_out, h_img, w_img)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Full Flow Matching model
# ---------------------------------------------------------------------------


class DiffusionTransformerFlowModel(ConditionalVectorField):
    """DiT-based conditional vector field u_t^θ(x|y) for Flow Matching.

    Architecture: Patchify → DiT (with time + label conditioning) → Depatchify.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        num_layers: int = 12,
        c: int = 1,
        dim: int = 256,
        heads: int = 4,
        final_dim: int = 10,
        n_classes: int = 11,
    ) -> None:
        super().__init__()

        # Time and label embedding
        self.time_embedder = FourierEncoder(dim)
        self.y_embedder = nn.Embedding(n_classes, dim)

        # Patchify image → tokens
        self.patchifier = Patchifier(
            img_size=img_size,
            patch_size=patch_size,
            c_in=c,
            dim=dim,
        )

        # Diffusion Transformer backbone
        n_tokens = (img_size // patch_size) ** 2
        self.dit = DiffusionTransformer(
            depth=num_layers,
            n_tokens=n_tokens,
            dim=dim,
            heads=heads,
        )

        # Depatchify tokens → image
        self.depatchifier = Depatchifier(
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            final_dim=final_dim,
            c_out=c,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate u_t^θ(x|y).

        Args:
            x: image, shape (b, c, h, w)
            t: time, shape (b,)
            y: class labels, shape (b,)
        Returns:
            predicted vector field, shape (b, c, h, w)
        """
        t_embed = self.time_embedder(t)  # (b, d)
        y_embed = self.y_embedder(y)  # (b, d)

        x = self.patchifier(x)  # (b, n, d)
        x = self.dit(x, t_embed + y_embed)  # (b, n, d)
        x = self.depatchifier(x)  # (b, c, h, w)
        return x
