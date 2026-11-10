from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.func import jacrev, vmap

from .sample import IsotropicGaussian, LabeledSampleable, Sampleable


class Alpha(ABC):
    """Scheduler for α_t in the Gaussian probability path p_t(x|z) = N(α_t·z, β_t²·I).

    Must satisfy: α₀ = 0, α₁ = 1.
    """

    def __init__(self) -> None:
        # Check α₀ = 0
        assert torch.allclose(self(torch.zeros(1)), torch.zeros(1)), (
            "Alpha must satisfy α₀ = 0"
        )
        # Check α₁ = 1
        assert torch.allclose(self(torch.ones(1)), torch.ones(1)), (
            "Alpha must satisfy α₁ = 1"
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate α_t.

        Args:
            t: shape (b,)
        Returns:
            α_t: shape (b,)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate d/dt α_t via automatic differentiation.

        Args:
            t: shape (b,)
        Returns:
            d/dt α_t: shape (b,)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)


class Beta(ABC):
    """Scheduler for β_t in the Gaussian probability path p_t(x|z) = N(α_t·z, β_t²·I).

    Must satisfy: β₀ = 1, β₁ = 0.
    """

    def __init__(self) -> None:
        # Check β₀ = 1
        assert torch.allclose(self(torch.zeros(1)), torch.ones(1)), (
            "Beta must satisfy β₀ = 1"
        )
        # Check β₁ = 0
        assert torch.allclose(self(torch.ones(1)), torch.zeros(1)), (
            "Beta must satisfy β₁ = 0"
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate β_t.

        Args:
            t: shape (b,)
        Returns:
            β_t: shape (b,)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate d/dt β_t via automatic differentiation.

        Args:
            t: shape (b,)
        Returns:
            d/dt β_t: shape (b,)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)



class LinearAlpha(Alpha):
    """α_t = t (linear schedule)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class LinearBeta(Beta):
    """β_t = 1 - t (linear schedule)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


class ConditionalProbabilityPath(nn.Module, ABC):
    """Abstract base class for conditional probability paths p_t(x|z).

    The marginal path is defined as p_t(x) = ∫ p_t(x|z) q(z) dz,
    where q(z) is the data distribution.
    """

    def __init__(self, p_simple: Sampleable, p_data: LabeledSampleable) -> None:
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """Sample from the marginal distribution p_t(x) = ∫ p_t(x|z) q(z) dz.

        Args:
            t: shape (b,)
        Returns:
            x: samples from p_t(x), shape (b, ...)
        """
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples)
        x = self.sample_conditional_path(z, t)
        return x

    @abstractmethod
    def sample_conditioning_variable(
        self, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the conditioning variable z and label y from q(z).

        Args:
            num_samples: the number of samples
        Returns:
            z: shape (b, ...)
            y: shape (b,)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from the conditional distribution p_t(x|z).

        Args:
            z: conditioning variable, shape (b, ...)
            t: time, shape (b,)
        Returns:
            x: samples from p_t(x|z), shape (b, ...)
        """
        pass

    @abstractmethod
    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the conditional vector field u_t(x|z).

        Args:
            x: shape (b, ...)
            z: shape (b, ...)
            t: shape (b,)
        Returns:
            u_t(x|z): shape (b, ...)
        """
        pass

    @abstractmethod
    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the conditional score ∇_x log p_t(x|z).

        Args:
            x: shape (b, ...)
            z: shape (b, ...)
            t: shape (b,)
        Returns:
            score: shape (b, ...)
        """
        pass


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """Gaussian conditional probability path: p_t(x|z) = N(α_t·z, β_t²·I).

    The conditional vector field is:
        u_t(x|z) = (α̇_t - β̇_t/β_t · α_t) · z + β̇_t/β_t · x
    """

    def __init__(
        self,
        p_data: Sampleable,
        p_simple_shape: list[int],
        alpha: Alpha,
        beta: Beta,
    ) -> None:
        p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        self.rearrange_scalar = Rearrange(f"b -> b{' 1' * len(p_simple_shape)}")

    def sample_conditioning_variable(
        self, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the conditioning variable z and label y from the data distribution.

        Args:
            num_samples: the number of samples
        Returns:
            z: shape (b, ...)
            y: shape (b,)
        """
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x ~ p_t(x|z) = N(α_t·z, β_t²·I).

        Args:
            z: shape (b, ...)
            t: shape (b,)
        Returns:
            x: shape (b, ...)
        """
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))
        return alpha_t * z + beta_t * torch.randn_like(z)

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate u_t(x|z) = (α̇_t - β̇_t/β_t · α_t) · z + β̇_t/β_t · x.

        Args:
            x: shape (b, c, h, w)
            z: shape (b, c, h, w)
            t: shape (b,)
        Returns:
            u_t(x|z): shape (b, c, h, w)
        """
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))
        dt_alpha_t = self.rearrange_scalar(self.alpha.dt(t))
        dt_beta_t = self.rearrange_scalar(self.beta.dt(t))

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate ∇_x log p_t(x|z) = (α_t·z - x) / β_t².

        Args:
            x: shape (b, ...)
            z: shape (b, ...)
            t: shape (b,)
        Returns:
            score: shape (b, ...)
        """
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))
        return (z * alpha_t - x) / beta_t**2
