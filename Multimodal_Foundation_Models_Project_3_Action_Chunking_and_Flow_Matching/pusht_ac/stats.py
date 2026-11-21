"""Paired statistics over common test reset seeds (new for this project)."""

from __future__ import annotations

import numpy as np


def paired_summary(a, b, n_boot: int = 20_000, seed: int = 0) -> dict:
    """Compare b - a on the same episodes. Returns the mean difference, a percentile
    bootstrap 95% CI over episodes, a two-sided sign-flip permutation p-value, and
    win / tie / loss counts."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    flips = rng.choice([-1.0, 1.0], (n_boot, len(d)))
    p = float(((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (n_boot + 1))
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "p_signflip": p,
            "wins": int((d > 1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()), "losses": int((d < -1e-9).sum())}
