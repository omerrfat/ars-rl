"""Summarise the offline CQL runs: a table (mean ± SD over seeds) and an overlay plot.

Usage:
    python summarise_offline.py

Reads results/offline/<env>/<policy>/s<seed>/{summary.json,eval.csv}, which
train_offline.py writes. For each env and dataset it prints, across seeds:
    final return   the 20-episode greedy test after the last update
    best return    the highest of the 20 periodic tests (10 episodes each) during training
    last mean Q    the average Q-value of the dataset's actions in the final logged update
and saves results/figures/offline_curves.png (one panel per env and dataset, one line per seed).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.config import get_config
from rl.paths import figures_dir, find_runs

ENVS = ["CartPole-v1", "Acrobot-v1"]
POLICIES = ["random", "medium"]
RESULTS = "results"


def load_runs(env, policy, root=RESULTS):
    """Return a list of (seed, summary dict, eval DataFrame), sorted by seed."""
    runs = []
    for seed, folder in find_runs("offline", env, policy, root):
        summary = json.load(open(os.path.join(folder, "summary.json")))
        runs.append((seed, summary, pd.read_csv(os.path.join(folder, "eval.csv"))))
    return runs


def mean_sd(values):
    # Sample SD (ddof=1) because 3 seeds are a sample, not the whole population.
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else float("nan")
    return values.mean(), sd


def main():
    print(f"{'env':<12} {'dataset':<7} {'seeds':>5}  {'final return':>18}  {'best return':>18}  {'last mean Q':>16}")
    fig, axes = plt.subplots(len(ENVS), len(POLICIES), figsize=(11, 7))

    for i, env in enumerate(ENVS):
        cfg = get_config(env)
        for j, policy in enumerate(POLICIES):
            runs = load_runs(env, policy)
            ax = axes[i][j]
            if not runs:
                ax.set_title(f"{env} / {policy}: no runs")
                continue

            finals = [s["final_return"] for _, s, _ in runs]
            bests = [s["best_eval"] for _, s, _ in runs]
            last_q = [df["q_mean"].iloc[-1] for _, _, df in runs]
            row = [mean_sd(v) for v in (finals, bests, last_q)]
            print(f"{env:<12} {policy:<7} {len(runs):>5}  "
                  f"{row[0][0]:>9.1f} ± {row[0][1]:>6.1f}  "
                  f"{row[1][0]:>9.1f} ± {row[1][1]:>6.1f}  "
                  f"{row[2][0]:>7.1f} ± {row[2][1]:>5.1f}")

            for seed, _, df in runs:
                ax.plot(df["gradient_steps"], df["eval_return"], label=f"seed {seed}")
            ax.axhline(cfg.target_return, ls="--", color="grey", lw=1, label="target")
            ax.set_title(f"{env} / {policy} data")
            ax.set_xlabel("Gradient updates")
            ax.set_ylabel("Greedy eval return")
            ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(figures_dir(RESULTS), "offline_curves.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
