"""Plotting helpers (new for this project): render a PushT state, T outlines,
loss curves, and shared figure styling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import make_env

PAL = {"mse": "#c26a2e", "flow": "#23686c", "A": "#7b8991", "B": "#8a6bb0", "C": "#23686c",
       "train": "#23686c", "val": "#c26a2e", "demo": "#9aa5ab"}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10, "axes.titlesize": 11})

_env = None


def _get_env(size=512):
    global _env
    if _env is None:
        _env = make_env(True, size)
    return _env


def render_state(state, size=512):
    """Image of the scene for a 5-d state [agent_x, agent_y, block_x, block_y, angle]."""
    env = _get_env(size)
    env.reset(options={"reset_to_state": np.asarray(state, dtype=np.float64)})
    env.unwrapped._last_action = None
    return env.render()


def block_polygons(state):
    """World-coordinate polygons of the T for a given state (pixel coords = world coords for a 512 render)."""
    env = _get_env()
    env.reset(options={"reset_to_state": np.asarray(state, dtype=np.float64)})
    blk = env.unwrapped.block
    return [np.array([blk.local_to_world(v) for v in s.get_vertices()]) for s in blk.shapes]


def draw_polys(ax, polys, scale=1.0, **kw):
    for p in polys:
        q = np.vstack([p, p[:1]]) * scale
        ax.plot(q[:, 0], q[:, 1], **kw)


def show_scene(ax, state, size=512, title=None):
    ax.imshow(render_state(state, size), extent=(0, 512, 512, 0))
    ax.set_xlim(0, 512); ax.set_ylim(512, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title)


def plot_losses(ax, log, title, ylabel="loss"):
    ax.plot(log.step, log.train_loss, color=PAL["train"], label="train (running mean)")
    ax.plot(log.step, log.val_loss, color=PAL["val"], label="validation episodes")
    ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
