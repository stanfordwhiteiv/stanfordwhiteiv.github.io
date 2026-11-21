"""PushT demonstration data: download, episode split, normalizer and chunk dataset.

Adapted from the Berkeley CS 185/285 (Spring 2026) HW1 starter file
``hw1_imitation/data.py``. Kept from the starter: ``Normalizer`` (mean/std with a
safe std), ``download_pusht``, ``load_pusht_zarr``, ``build_valid_indices`` and the
sliding-window ``PushtChunkDataset``.

Changes for this project:
  * ``split_episodes``: split by complete episode with a recorded seed.
  * ``Normalizer.from_episodes`` fits statistics on training episodes only, and
    ``save`` / ``load`` keep the statistics frozen across Parts 1-3.
  * ``PushtChunkDataset`` accepts a list of episode ids, so chunks are only built
    inside the chosen episodes, and pre-computes all normalized tensors.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    @classmethod
    def from_episodes(
        cls, states: np.ndarray, actions: np.ndarray, episode_ends: np.ndarray, episode_ids
    ) -> "Normalizer":
        """Fit on the time steps of the given (training) episodes only."""
        rows = np.concatenate([np.arange(*episode_bounds(episode_ends, e)) for e in episode_ids])
        return cls.from_data(states[rows], actions[rows])

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({k: getattr(self, k).tolist() for k in
                                          ("state_mean", "state_std", "action_mean", "action_std")}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        d = json.loads(Path(path).read_text())
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


def download_pusht(dataset_dir: Path) -> Path:
    """Download and extract the Push-T dataset if needed; return the Zarr path."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path
    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def episode_bounds(episode_ends: np.ndarray, e: int) -> tuple[int, int]:
    start = 0 if e == 0 else int(episode_ends[e - 1])
    return start, int(episode_ends[e])


def split_episodes(num_episodes: int, val_fraction: float = 0.2, seed: int = 0):
    """Random split of complete episodes into (train_ids, val_ids)."""
    perm = np.random.default_rng(seed).permutation(num_episodes)
    n_val = int(round(val_fraction * num_episodes))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int, episode_ids=None) -> np.ndarray:
    """Start indices t such that [t, t+K) lies inside one episode."""
    if episode_ids is None:
        episode_ids = range(len(episode_ends))
    indices: list[int] = []
    for e in episode_ids:
        start, end = episode_bounds(episode_ends, e)
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class PushtChunkDataset(Dataset):
    """(normalized state, normalized action chunk) pairs from a sliding window
    that never crosses an episode boundary."""

    def __init__(self, states, actions, episode_ends, chunk_size, normalizer=None, episode_ids=None):
        self.chunk_size = chunk_size
        self.indices = build_valid_indices(episode_ends, chunk_size, episode_ids)
        s = states[self.indices]
        a = np.stack([actions[t: t + chunk_size] for t in self.indices])
        if normalizer is not None:
            s = normalizer.normalize_state(s)
            a = normalizer.normalize_action(a)
        self.obs = torch.as_tensor(s, dtype=torch.float32)
        self.chunks = torch.as_tensor(a, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.chunks[idx]
