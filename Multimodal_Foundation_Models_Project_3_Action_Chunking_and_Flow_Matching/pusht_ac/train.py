"""Training loops.

``train_policy`` fills in the ``### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###`` of
the Berkeley starter ``hw1_imitation/train.py`` (Adam, shuffled minibatches,
periodic evaluation). Changes: validation imitation loss on held-out episodes,
validation rollouts on the validation reset seeds, checkpoint selection on
validation only, CSV logging instead of W&B.

``finetune_flow`` is new (Part 3): reward-weighted / equal-weight flow-matching
fine-tuning with demonstration replay and matched minibatches, noise and times.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch

from .evaluation import VAL_SEEDS, evaluate, summarize


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def dataset_loss(policy, obs, chunks, seed: int = 123, batch: int = 4096) -> float:
    """Mean per-chunk loss on a whole dataset. Flow noise/times use a fixed generator
    so the validation curve is not jittered by resampling."""
    policy.eval()
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for i in range(0, len(obs), batch):
        tot += policy.per_example_loss(obs[i:i + batch], chunks[i:i + batch], generator=g).sum().item()
    return tot / len(obs)


def train_policy(policy, train_ds, val_ds, normalizer, *, steps=40_000, batch_size=128, lr=3e-4,
                 eval_every=2_000, log_every=200, val_seeds=VAL_SEEDS, num_steps=10, seed=0,
                 rollout_offset=0, verbose=True):
    """Returns (best_policy, log DataFrame, eval DataFrame, info dict).
    The best checkpoint maximizes the mean validation-rollout score; ties -> lower val loss."""
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(train_ds)
    log, evals = [], []
    best, best_key = None, None
    run_loss, t_train, t_eval = [], 0.0, 0.0
    t_start = time.perf_counter()
    perm, pos = torch.randperm(n, generator=g), 0
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        policy.train()
        if pos + batch_size > n:  # new epoch, drop the last partial batch
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos: pos + batch_size]
        pos += batch_size
        loss = policy.compute_loss(train_ds.obs[idx], train_ds.chunks[idx], generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss.append(loss.item())
        t_train += time.perf_counter() - t0
        if step % log_every == 0:
            log.append({"step": step, "epoch": step * batch_size / n, "train_loss": float(np.mean(run_loss)),
                        "val_loss": dataset_loss(policy, val_ds.obs, val_ds.chunks)})
            run_loss = []
        if step % eval_every == 0:
            t0 = time.perf_counter()
            df, _ = evaluate(policy, normalizer, val_seeds, policy_seed_offset=rollout_offset, num_steps=num_steps)
            t_eval += time.perf_counter() - t0
            s = summarize(df)
            vl = log[-1]["val_loss"]
            evals.append({"step": step, "val_score": s["score_mean"], "val_success": s["success_rate"], "val_loss": vl})
            key = (s["score_mean"], -vl)
            if best_key is None or key > best_key:
                best_key, best = key, (step, copy.deepcopy(policy.state_dict()))
            if verbose:
                print(f"step {step:6d}  train {log[-1]['train_loss']:.4f}  val {vl:.4f}  "
                      f"val-rollout score {s['score_mean']:.3f}  success {s['success_rate']:.2f}", flush=True)
    final = copy.deepcopy(policy)
    policy.load_state_dict(best[1])
    info = {"best_step": best[0], "train_time_s": t_train, "val_rollout_time_s": t_eval,
            "wall_time_s": time.perf_counter() - t_start}
    return policy.eval(), pd.DataFrame(log), pd.DataFrame(evals), info, final


def reward_weights(scores: np.ndarray, beta: float, cap: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """u_j = min(exp(beta R_j), cap); w_j = u_j / mean(u). Computed once, held constant."""
    u = np.minimum(np.exp(beta * np.asarray(scores, dtype=np.float64)), cap)
    return u, u / u.mean()


def finetune_flow(base_policy, new_obs, new_chunks, weights, demo_ds, *, updates=300, lr=1e-4,
                  batch_new=64, batch_demo=64, seed=0, mix=0.5, eval_every=None, normalizer=None,
                  val_seeds=VAL_SEEDS, val_ds=None):
    """L = mix * mean_j(w_j * l_FM(o_j, A_j)) + (1 - mix) * mean(l_FM on demos).

    A fresh copy and a fresh Adam optimizer are made here. All randomness (collected
    indices, demo indices, eps, tau) comes from one generator seeded with ``seed``,
    so two branches with the same seed see identical minibatches, noise and times.
    """
    policy = copy.deepcopy(base_policy).train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    w = torch.as_tensor(weights, dtype=torch.float32)
    m, nd = len(new_obs), len(demo_ds)
    k, d = policy.chunk_size, policy.action_dim
    log, evals = [], []
    t_train = 0.0
    for step in range(1, updates + 1):
        t0 = time.perf_counter()
        policy.train()
        i_new = torch.randint(m, (batch_new,), generator=g)          # uniform, with replacement
        i_demo = torch.randint(nd, (batch_demo,), generator=g)
        eps_new = torch.randn((batch_new, k, d), generator=g)
        tau_new = torch.rand(batch_new, generator=g)
        eps_demo = torch.randn((batch_demo, k, d), generator=g)
        tau_demo = torch.rand(batch_demo, generator=g)
        per_chunk = policy.per_example_loss(new_obs[i_new], new_chunks[i_new], noise=eps_new, tau=tau_new)
        loss_new = (w[i_new].detach() * per_chunk).mean()   # weight each chunk's own error
        loss_demo = policy.per_example_loss(demo_ds.obs[i_demo], demo_ds.chunks[i_demo],
                                            noise=eps_demo, tau=tau_demo).mean()
        loss = mix * loss_new + (1 - mix) * loss_demo
        opt.zero_grad()
        loss.backward()
        opt.step()
        t_train += time.perf_counter() - t0
        row = {"step": step, "loss": loss.item(), "loss_new_weighted": loss_new.item(),
               "loss_new_unweighted": per_chunk.mean().item(), "loss_demo": loss_demo.item()}
        if val_ds is not None and (step % 25 == 0 or step == 1):
            row["val_demo_loss"] = dataset_loss(policy, val_ds.obs, val_ds.chunks)
        log.append(row)
        if eval_every and step % eval_every == 0:
            df, _ = evaluate(policy, normalizer, val_seeds)
            evals.append({"step": step, **summarize(df)})
    return policy.eval(), pd.DataFrame(log), pd.DataFrame(evals), {"finetune_time_s": t_train}
