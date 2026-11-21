Pilot run before the main experiments (validation seeds 0-49 only; no test seeds used).
Both policies, 3 hidden layers of width 128 or 256, 60k Adam steps (lr 3e-4, batch 128), validation rollouts every 3k steps.
Best validation score: MSE 128 -> 0.53, MSE 256 -> 0.58, flow 128 -> 0.53, flow 256 -> 0.81 (flow 128 was still improving).
Decision: width 256 (the Berkeley starter default) and 150k steps for both policies in Parts 1-2.
