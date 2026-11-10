from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .simulator import ODE


class ConditionalVectorField(nn.Module, ABC):
    """Conditional vector field u_t^θ(x|y) — the learnable component.

    Subclasses implement the forward pass that takes (x, t, y) and returns
    the predicted vector field.
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate u_t^θ(x|y).

        Args:
            x: shape (b, ...)
            t: shape (b,)
            y: shape (b,)
        Returns:
            u_t^θ(x|y): shape (b, ...)
        """
        pass


class CFGVectorFieldODE(ODE):
    """Classifier-Free Guidance (CFG) wrapper for a conditional vector field.

    Implements: v_cfg = (1 - w) · u_t^θ(x|∅) + w · u_t^θ(x|y)
    where w is the guidance scale and ∅ is the null label.
    """

    def __init__(
        self,
        net: ConditionalVectorField,
        null_label: int,
        guidance_scale: float = 1.0,
    ) -> None:
        self.net = net
        self.guidance_scale = guidance_scale
        self.null_label = null_label

    def drift_coefficient(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute CFG-adjusted drift: (1-w)·v(x,t,∅) + w·v(x,t,y).

        Args:
            x: shape (b, ...)
            t: shape (b,)
            y: shape (b,)
        Returns:
            drift: shape (b, ...)
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.full_like(y, self.null_label)
        unguided_vector_field = self.net(x, t, unguided_y)
        return (
            1 - self.guidance_scale
        ) * unguided_vector_field + self.guidance_scale * guided_vector_field
