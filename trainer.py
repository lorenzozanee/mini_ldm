from __future__ import annotations

import csv
import os
import random
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .path import GaussianConditionalProbabilityPath
from .sample import LatentMNISTSampler, MNISTSampler
from .utils import MiB, model_size_b, save_mnist_grid, visualize_output


class Trainer(ABC):
    """Abstract base class for Flow Matching trainers.

    Provides a standard training loop with linear warmup, checkpointing,
    and loss tracking. Subclasses implement ``get_train_loss``.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.model: nn.Module | None = None
        self.opt: torch.optim.Optimizer | None = None
        self.output_dir: str | None = None

    @abstractmethod
    def get_train_loss(self, **kwargs: object) -> torch.Tensor:
        """Compute the training loss for one batch.

        Returns:
            scalar loss tensor
        """
        pass

    def checkpoint(self, step: int) -> None:
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            },
            os.path.join(self.output_dir, f"step_{step:06d}.pt"),
        )

    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    @staticmethod
    def random_name() -> str:
        adjectives = [
            "autumn",
            "hidden",
            "bitter",
            "misty",
            "silent",
            "empty",
            "dry",
            "dark",
            "summer",
            "icy",
            "delicate",
            "quiet",
            "white",
            "cool",
            "spring",
            "winter",
            "patient",
        ]
        foods = [
            "apple",
            "banana",
            "pear",
            "plum",
            "orange",
            "persimmon",
            "tangerine",
            "durian",
            "jackfruit",
            "jicama",
            "cantaloupe",
            "watermelon",
            "peach",
        ]
        return (
            f"{random.choice(adjectives)}-{random.choice(foods)}-{uuid.uuid4().hex[:8]}"
        )

    def train(
        self,
        model: nn.Module,
        num_steps: int,
        lr: float = 1e-3,
        warmup_steps: int = 500,
        ckpt_every: int | None = 500,
        run_name: str | None = None,
        output_root: str | Path = "runs",
        **kwargs: object,
    ) -> tuple[list[float], list[int]]:
        """Run the training loop with linear warmup.

        Args:
            model: the ConditionalVectorField to train
            num_steps: total training steps
            lr: peak learning rate (after warmup)
            warmup_steps: linear warmup from 0 to lr
            ckpt_every: save checkpoint every N steps (None to disable)
            run_name: name for the output directory (auto-generated if None)
            **kwargs: forwarded to ``get_train_loss`` (e.g., ``batch_size``)

        Returns:
            losses: list of per-step loss values
            steps: list of step indices
        """
        if num_steps < 1 or warmup_steps < 0 or lr <= 0:
            raise ValueError("num_steps and lr must be positive; warmup_steps must be nonnegative")
        if ckpt_every is not None and ckpt_every < 1:
            raise ValueError("ckpt_every must be positive or None")
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join(output_root, run_name)
        os.makedirs(self.output_dir, exist_ok=False)
        print(f"Initialized output directory at: {self.output_dir}")

        # Model size
        self.model = model
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        # Optimizer
        self.opt = self.get_optimizer(lr)
        self.model.train()

        for pg in self.opt.param_groups:
            pg["lr"] = 0.0

        # Main training loop
        losses: list[float] = []
        steps: list[int] = []

        with open(os.path.join(self.output_dir, "losses.csv"), "w", newline="") as log:
            writer = csv.writer(log)
            writer.writerow(["step", "loss", "lr"])
            pbar = tqdm(range(1, num_steps + 1), mininterval=5)
            for step in pbar:
                cur_lr = lr * min(step / max(warmup_steps, 1), 1.0)
                for pg in self.opt.param_groups:
                    pg["lr"] = cur_lr

                self.opt.zero_grad(set_to_none=True)
                loss = self.get_train_loss(**kwargs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step}")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                self.opt.step()

                value = loss.detach().item()
                losses.append(value)
                steps.append(step)
                writer.writerow([step, value, cur_lr])
                pbar.set_description(f"Step {step}, lr={cur_lr:.2e}, loss={value:.4f}", refresh=False)

                if (ckpt_every is not None and step % ckpt_every == 0) or step == num_steps:
                    log.flush()
                    self.model.eval()
                    self.checkpoint(step)
                    self.model.train()

        self.model.eval()
        return losses, steps


class CFGTrainer(Trainer):
    """Trainer with Classifier-Free Guidance (CFG) loss.

    During training, each label is independently replaced with the null label
    with probability ``eta``, so the model learns both conditional and
    unconditional vector fields.
    """

    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        eta: float,
        null_label: int,
        eps: float = 0.001,
        **kwargs: object,
    ) -> None:
        if not (0 < eta < 1):
            raise ValueError(f"eta must be in (0, 1), got {eta}")
        if not 0 < eps < 1:
            raise ValueError("eps must be in (0, 1)")
        super().__init__(**kwargs)
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z, y from data distribution
        z, y = self.path.p_data.sample(batch_size)

        # Step 2: CFG — replace each label with null_label w.p. eta
        xi = torch.rand(y.shape[0], device=y.device)
        y = torch.where(xi < self.eta, self.null_label, y)

        # Step 3: Sample t ∈ [0, 1-eps] and x ~ p_t(x|z)
        t = torch.rand(batch_size, device=z.device) * (1 - self.eps)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Flow Matching loss = ||u_t^θ(x|y) - u_t(x|z)||²
        ut_theta = self.model(x, t, y)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        return torch.square(ut_theta - ut_ref).mean()


class MNISTCFGTrainer(CFGTrainer):
    """CFG Trainer with MNIST-specific checkpoint saving and visualization."""

    def checkpoint(self, step: int) -> None:
        super().checkpoint(step)
        visualize_output(
            self.model,
            self.path,
            save_path=os.path.join(self.output_dir, f"step_{step:06d}_output.png"),
            use_tqdm=False,
        )


class MNISTVAETrainer(Trainer):
    """Train the notebook VAE and save input/reconstruction pairs."""

    def __init__(self, mnist: MNISTSampler):
        super().__init__()
        self.mnist = mnist

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        x, _ = self.mnist.sample(batch_size)
        return self.model.compute_loss(*self.model(x), x)

    @torch.no_grad()
    def checkpoint(self, step: int) -> None:
        super().checkpoint(step)
        x, _ = self.mnist.sample(10)
        _, _, recon, _ = self.model(x)
        save_mnist_grid(
            torch.cat([x, recon]),
            os.path.join(self.output_dir, f"step_{step:06d}_reconstruction.png"),
            nrow=10,
        )


class LatentCFGTrainer(CFGTrainer):
    """Reuse the CFG objective with frozen VAE posterior samples."""

    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        eta: float = 0.35,
        null_label: int = 10,
        sample_steps: int = 100,
        samples_per_class: int = 4,
    ):
        if not isinstance(path.p_data, LatentMNISTSampler):
            raise TypeError("LatentCFGTrainer requires a LatentMNISTSampler")
        super().__init__(path=path, eta=eta, null_label=null_label)
        self.sample_steps = sample_steps
        self.samples_per_class = samples_per_class

    def checkpoint(self, step: int) -> None:
        super().checkpoint(step)
        visualize_output(
            self.model,
            self.path,
            vae=self.path.p_data.vae,
            samples_per_class=self.samples_per_class,
            num_timesteps=self.sample_steps,
            guidance_scales=[3.0],
            null_label=self.null_label,
            save_path=os.path.join(self.output_dir, f"step_{step:06d}_samples.png"),
        )
