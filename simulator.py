from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class ODE(ABC):
    """Ordinary differential equation dx/dt = v(x, t)."""

    @abstractmethod
    def drift_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Returns the drift coefficient v(x_t, t).

        Args:
            xt: shape (b, ...)
            t: shape (b,)
        Returns:
            drift_coefficient: shape (b, ...)
        """
        pass




class Simulator(ABC):
    """Abstract base class for numerical SDE/ODE solvers."""

    @abstractmethod
    def step(
        self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Take one simulation step.

        Args:
            xt: shape (b, ...)
            t: shape (b,)
            dt: shape (b,)
        Returns:
            next_xt: shape (b, ...)
        """
        pass

    @torch.no_grad()
    def simulate(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        use_tqdm: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Simulate using the discretization given by ts.

        Args:
            x: initial state, shape (b, ...)
            ts: time grid, shape (b, nt)
        Returns:
            x_final: final state, shape (b, ...)
        """
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        use_tqdm: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Simulate and record full trajectory.

        Args:
            x: initial state, shape (b, ...)
            ts: time grid, shape (b, nt)
        Returns:
            x_traj: full trajectory, shape (b, nt, ...)
        """
        x_traj = [x.clone()]
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            x_traj.append(x.clone())
        return torch.stack(x_traj, dim=1)


class EulerSimulator(Simulator):
    """First-order Euler method for ODEs: x_{t+h} = x_t + v(x_t, t) · h."""

    def __init__(self, ode: ODE) -> None:
        super().__init__()
        self.ode = ode

    def step(
        self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        h = h.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h


