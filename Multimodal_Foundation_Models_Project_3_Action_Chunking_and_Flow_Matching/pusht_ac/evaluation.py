"""Simulator rollouts for PushT chunk policies.

Adapted from ``evaluate_policy`` in the Berkeley HW1 starter file
``hw1_imitation/evaluation.py``. Kept: the receding-horizon execution loop
(normalize obs -> sample chunk -> denormalize -> clip to action bounds -> execute
in order -> re-query), and the episode-maximum reward metric.

Changes for this project:
  * reset seeds are passed in as a list (validation / test / collection groups)
    instead of the fixed ``range(100)``;
  * every episode has a recorded policy-noise seed (a ``torch.Generator``) so
    stochastic flow policies are reproducible;
  * the loop stops the moment ``terminated or truncated`` is reported, even in
    the middle of a chunk;
  * success (``info["is_success"]``), episode length, runtime and the reward
    trace are recorded; optional frame capture writes GIFs instead of W&B videos;
  * ``record=True`` stores every policy query (starting obs + the executed,
    denormalized, clipped actions) for the Part 3 collection dataset;
  * W&B logging and checkpoint artifacts are removed (local CSV logs only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import gym_pusht  # noqa: F401  (registers the environment)
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image

ENV_ID = "gym_pusht/PushT-v0"

VAL_SEEDS = list(range(0, 50))
TEST_SEEDS = list(range(1000, 1100))
COLLECT_SEEDS = list(range(2000, 2025))


@dataclass
class Episode:
    reset_seed: int
    policy_seed: int
    max_reward: float
    success: bool
    length: int
    rewards: list = field(default_factory=list)
    queries: list = field(default_factory=list)   # (obs, executed actions [k,2], complete?)
    frames: list = field(default_factory=list)
    policy_time: float = 0.0
    total_time: float = 0.0


def make_env(render: bool = False, size: int = 256):
    return gym.make(ENV_ID, obs_type="state", render_mode="rgb_array",
                    visualization_width=size, visualization_height=size)


def rollout(policy, normalizer, reset_seed: int, policy_seed: int = 0, *, num_steps: int = 10,
            env=None, render: bool = False, record: bool = False, frame_size: int = 256) -> Episode:
    own_env = env is None
    env = env or make_env(render, frame_size)
    low, high = env.action_space.low, env.action_space.high
    k = policy.chunk_size
    gen = torch.Generator().manual_seed(int(policy_seed))
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=int(reset_seed))
    ep = Episode(int(reset_seed), int(policy_seed), 0.0, False, 0)
    if render:
        ep.frames.append(env.render())
    done = False
    policy.eval()
    while not done:
        q_obs = obs.copy()
        tp = time.perf_counter()
        with torch.no_grad():
            s = torch.as_tensor(normalizer.normalize_state(obs), dtype=torch.float32).unsqueeze(0)
            chunk = policy.sample_actions(s, num_steps=num_steps, generator=gen)[0].numpy()
        ep.policy_time += time.perf_counter() - tp
        chunk = np.clip(normalizer.denormalize_action(chunk), low, high)
        executed = []
        for a in chunk:
            obs, r, terminated, truncated, info = env.step(a.astype(np.float32))
            executed.append(a.copy())
            ep.rewards.append(float(r))
            ep.success |= bool(info.get("is_success", False))
            if render:
                ep.frames.append(env.render())
            done = terminated or truncated
            if done:
                break
        if record:
            ep.queries.append((q_obs, np.asarray(executed, dtype=np.float32), len(executed) == k))
    ep.length = len(ep.rewards)
    ep.max_reward = float(max(ep.rewards))
    ep.total_time = time.perf_counter() - t0
    if own_env:
        env.close()
    return ep


def evaluate(policy, normalizer, reset_seeds, *, policy_seed_offset: int = 0, num_steps: int = 10,
             record: bool = False):
    """Run one episode per reset seed. The policy seed is reset_seed + offset.
    Returns (per-episode DataFrame, list of Episode objects)."""
    env = make_env()
    eps = [rollout(policy, normalizer, s, s + policy_seed_offset, num_steps=num_steps, env=env, record=record)
           for s in reset_seeds]
    env.close()
    df = pd.DataFrame([{"reset_seed": e.reset_seed, "policy_seed": e.policy_seed, "max_reward": e.max_reward,
                        "success": e.success, "length": e.length, "policy_time_s": e.policy_time,
                        "total_time_s": e.total_time} for e in eps])
    return df, eps


def summarize(df: pd.DataFrame) -> dict:
    return {"score_mean": df.max_reward.mean(), "score_std": df.max_reward.std(ddof=1),
            "success_rate": df.success.mean(), "mean_length": df.length.mean(),
            "rollout_time_s": df.total_time_s.sum(), "n": len(df)}


def save_gif(frames, path, fps: int = 20, size: int = 256, stride: int = 1) -> None:
    """Control rate is 10 Hz, so fps=20 plays at 2x speed."""
    imgs = [np.asarray(Image.fromarray(f).resize((size, size), Image.BILINEAR)) for f in frames[::stride]]
    imgs += [imgs[-1]] * 10  # brief hold on the final frame
    imageio.mimsave(path, imgs, duration=1000 / fps, loop=0)
