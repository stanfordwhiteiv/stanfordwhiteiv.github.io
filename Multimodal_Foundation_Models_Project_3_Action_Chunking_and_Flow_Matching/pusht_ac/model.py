"""MSE and conditional flow-matching action-chunk policies.

The class layout (``BasePolicy`` with ``compute_loss`` / ``sample_actions``,
``MSEPolicy``, ``FlowMatchingPolicy``, ``build_policy``) follows the Berkeley HW1
starter file ``hw1_imitation/model.py``; the starter left both policies as TODOs.
Everything inside the policies is implemented here. Additions: ``per_example_loss``
(one loss value per chunk, needed for reward weighting in Part 3), optional
explicit noise / flow times / torch generators so that runs can be matched, and
``save_policy`` / ``load_policy``.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from torch import nn


def mlp(in_dim: int, out_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dims = tuple(hidden_dims)

    @abc.abstractmethod
    def per_example_loss(self, state: torch.Tensor, action_chunk: torch.Tensor, **kw) -> torch.Tensor:
        """Loss for every chunk in the batch, shape (batch,)."""

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor, **kw) -> torch.Tensor:
        return self.per_example_loss(state, action_chunk, **kw).mean()

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor, *, num_steps: int = 10,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """f_theta(o) in R^{K x 2}, trained with (1/2K)||f_theta(o) - A||_F^2."""

    def __init__(self, state_dim, action_dim, chunk_size, hidden_dims=(128, 128, 128)) -> None:
        super().__init__(state_dim, action_dim, chunk_size, hidden_dims)
        self.net = mlp(state_dim, chunk_size * action_dim, self.hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).view(-1, self.chunk_size, self.action_dim)

    def per_example_loss(self, state, action_chunk, **kw):
        # mean over the K*2 coordinates == (1/2K) * squared Frobenius norm
        return (self(state) - action_chunk).square().flatten(1).mean(1)

    def sample_actions(self, state, *, num_steps=10, generator=None):
        return self(state)  # deterministic: num_steps / generator are ignored


class FlowMatchingPolicy(BasePolicy):
    """Velocity field v_theta(o, A_tau, tau) in R^{K x 2}.

    Convention: tau = 0 is Gaussian noise, tau = 1 is an action chunk.
        A_tau = (1 - tau) * eps + tau * A,   target u = A - eps.
    Input is the concatenation [o (5), A_tau (2K), tau (1)] = 22 values for K = 8.
    """

    def __init__(self, state_dim, action_dim, chunk_size, hidden_dims=(128, 128, 128)) -> None:
        super().__init__(state_dim, action_dim, chunk_size, hidden_dims)
        self.net = mlp(state_dim + chunk_size * action_dim + 1, chunk_size * action_dim, self.hidden_dims)

    def velocity(self, state: torch.Tensor, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, z.flatten(1), tau.view(-1, 1)], dim=1)
        return self.net(x).view(-1, self.chunk_size, self.action_dim)

    @staticmethod
    def interpolate(action_chunk, noise, tau):
        t = tau.view(-1, 1, 1)
        return (1 - t) * noise + t * action_chunk, action_chunk - noise

    def per_example_loss(self, state, action_chunk, noise=None, tau=None, generator=None, **kw):
        b = action_chunk.shape[0]
        if noise is None:
            noise = torch.randn(action_chunk.shape, generator=generator, device=action_chunk.device)
        if tau is None:
            tau = torch.rand(b, generator=generator, device=action_chunk.device)
        a_tau, target = self.interpolate(action_chunk, noise, tau)
        pred = self.velocity(state, a_tau, tau)
        return (pred - target).square().flatten(1).mean(1)

    @torch.no_grad()
    def sample_actions(self, state, *, num_steps=10, generator=None, z0=None):
        """Forward Euler from tau=0 (noise) to tau=1; the observation is held fixed."""
        b = state.shape[0]
        z = z0 if z0 is not None else torch.randn(
            (b, self.chunk_size, self.action_dim), generator=generator, device=state.device)
        for n in range(num_steps):
            tau = torch.full((b,), n / num_steps, device=state.device)
            z = z + self.velocity(state, z, tau) / num_steps
        return z


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(policy_type: PolicyType, *, state_dim=5, action_dim=2, chunk_size=8,
                 hidden_dims=(128, 128, 128)) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(state_dim, action_dim, chunk_size, hidden_dims)
    if policy_type == "flow":
        return FlowMatchingPolicy(state_dim, action_dim, chunk_size, hidden_dims)
    raise ValueError(f"Unknown policy type: {policy_type}")


def save_policy(policy: BasePolicy, path: Path, **meta) -> None:
    kind = "mse" if isinstance(policy, MSEPolicy) else "flow"
    torch.save({"policy_type": kind, "state_dim": policy.state_dim, "action_dim": policy.action_dim,
                "chunk_size": policy.chunk_size, "hidden_dims": policy.hidden_dims,
                "state_dict": policy.state_dict(), "meta": meta}, path)


def load_policy(path: Path) -> BasePolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    p = build_policy(ck["policy_type"], state_dim=ck["state_dim"], action_dim=ck["action_dim"],
                     chunk_size=ck["chunk_size"], hidden_dims=tuple(ck["hidden_dims"]))
    p.load_state_dict(ck["state_dict"])
    p.meta = ck.get("meta", {})
    return p.eval()
