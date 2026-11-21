# Multimodal Foundation Models Project 3 - Action Chunking and Reward-Guided Flow Matching

PushT (`gym_pusht/PushT-v0`, state observations) with action chunks of length K = 8:

- **Part 1:** direct MSE regression of action chunks.
- **Part 2:** conditional flow matching with a 10-step Euler sampler.
- **Part 3:** reward-weighted vs. equal-weight flow-matching fine-tuning on the policy's own rollouts (A / B / C study).

## Starter code and what changed

Parts 1 and 2 build on the **UC Berkeley CS 185/285, Spring 2026, Assignment 1 (Imitation Learning)** starter code:
<https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw1>. The adapted files are in `pusht_ac/`,
and each module's docstring lists exactly what was kept and what was changed. In short:

| File | From the starter | Changed / added here |
|---|---|---|
| `pusht_ac/data.py` | `Normalizer`, `download_pusht`, `load_pusht_zarr`, `build_valid_indices`, `PushtChunkDataset` | episode-level train/val split; normalizer fitted on training episodes only, with save/load; datasets restricted to chosen episodes |
| `pusht_ac/model.py` | `BasePolicy` / `MSEPolicy` / `FlowMatchingPolicy` / `build_policy` skeleton (policies were TODOs) | both policies, their losses and the Euler sampler; per-chunk losses; seeded generators; save/load |
| `pusht_ac/train.py` | the training-loop TODO in `train.py` | training loop with validation loss and validation rollouts for checkpoint selection; Part 3 weighted / equal-weight fine-tuning with demo replay |
| `pusht_ac/evaluation.py` | the receding-horizon loop of `evaluate_policy` | reset-seed lists, recorded policy-noise seeds, stop mid-chunk on termination, success / length / runtime logging, rollout recording for Part 3, GIFs; no W&B |
| `pusht_ac/viz.py`, `pusht_ac/stats.py` | n/a | new: plotting and paired bootstrap / sign-flip statistics |

## Dependencies

```
pip install torch gym-pusht "pymunk<7" zarr numpy pandas matplotlib imageio imageio-ffmpeg pillow jupyter
```

(or `pip install -r requirements.txt`). Tested with Python 3.12.10, torch 2.15.0+cpu, gym-pusht 0.1.8, gymnasium 1.4.0
and pymunk 6.11.1 on a laptop CPU (Intel, 4 torch threads). No GPU is needed.

## Data

On first run, `part1.ipynb` downloads the PushT demonstrations (`pusht.zip`, ~31 MB, the same URL as the starter) to
`data/` and extracts `pusht/pusht_cchi_v7_replay.zarr` (206 episodes, 25,650 steps).

## Protocol

| Item | Value |
|---|---|
| Demonstration split | by complete episode, 80/20, seed 0 → 165 train / 41 val episodes (`checkpoints/split.json`) |
| Normalizer | mean/std of training episodes only (`checkpoints/normalizer.json`), reused unchanged in all parts |
| Networks | MLP 3 × 256, ReLU; MSE: 5 → 16; flow: 22 → 16 |
| Training | Adam 3e-4, batch 128, 150k steps, seed 0 |
| Checkpoint selection | best mean validation-rollout score on reset seeds 0–49 (validation only) |
| Final test | reset seeds 1000–1099 (100 episodes), policy-noise seed = reset seed (+ two repeat passes with offsets 10000 / 20000) |
| Part 3 collection | reset seeds 2000–2024 × 2 policy-noise seeds = 50 episodes |
| Score | mean over episodes of $R_e=\max_t r_{e,t}$; success = `info["is_success"]` reported separately |

## Notebooks

| Notebook | Content |
|---|---|
| `part1.ipynb` | data inspection, MSE policy training, conditional-mean illustration, test evaluation, rollouts |
| `part2.ipynb` | flow-matching shape checks, training, Euler sampling, MSE vs. flow comparison, N sweep, sampled chunks |
| `part3.ipynb` | collection with the frozen Part 2 policy, weights, validation budget check, B/C fine-tuning (3 seeds), A/B/C test comparison |

Run them in order. Training is skipped when a checkpoint already exists in `checkpoints/`; delete the checkpoint to
retrain. Results (CSV/JSON logs, figures, GIFs) go to `results_part1/`, `results_part2/` and `results_part3/`. GIFs
play at 2× speed (10 Hz control, 20 fps). Measured wall-clock times on the test machine: Part 1 ≈ 7 min, Part 2 ≈ 7 min,
Part 3 ≈ 6 min.

## Headline results (test seeds 1000–1099)

| Policy | Score (mean ± std) | Success |
|---|---|---|
| MSE (Part 1) | 0.575 ± 0.397 | 11% |
| Flow matching, N = 10 (Part 2 = condition A) | 0.840 ± 0.265 | 22% |
| B: equal-weight fine-tuning (3 seeds) | 0.782 – 0.815 | 14 – 22% |
| C: reward-weighted fine-tuning (3 seeds) | 0.766 – 0.801 | 14 – 21% |

C − B = −0.013 (95% CI −0.037 to +0.010): no detectable effect of reward weighting. The collection scores were so high
that 89% of chunks hit the weight cap.

The report is `index.html`.
