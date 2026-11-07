"""Quick learning-curve plot: greedy evaluation return against online env steps.

Usage:
    python plot.py --env CartPole-v1                 # online + O2O runs found in results/
    python plot.py --env CartPole-v1 --modes online

(summarise.py makes the full set of figures; this is the quick single-env version.)
Runs with several seeds are averaged; the shaded band is the min-max range across seeds.
Offline runs are skipped: their x-axis is gradient updates, not online env steps.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl.config import get_config
from rl.paths import figures_dir, find_runs

# (label, mode, dataset) for every curve that has online env steps on its x-axis
CURVES = [("online", "online", None), ("o2o-random", "o2o", "random"), ("o2o-medium", "o2o", "medium")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--modes", nargs="*", default=None, help="labels to plot, e.g. online o2o-medium")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    cfg = get_config(args.env)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, mode, dataset in CURVES:
        if args.modes and label not in args.modes:
            continue
        runs = [pd.read_csv(os.path.join(folder, "eval.csv"))
                for _, folder in find_runs(mode, args.env, dataset, args.results)]
        if not runs:
            continue
        n = min(len(r) for r in runs)                 # align runs to the shortest
        steps = runs[0]["env_steps"].to_numpy()[:n]
        returns = np.stack([r["eval_return"].to_numpy()[:n] for r in runs])
        ax.plot(steps, returns.mean(axis=0), label=f"{label} (mean of {len(runs)} seeds)")
        # min-max band: unlike mean +/- std it never extends past scores the env can give
        ax.fill_between(steps, returns.min(axis=0), returns.max(axis=0), alpha=0.2)
    if not ax.lines:
        raise SystemExit(f"No online/o2o runs for {args.env} in {args.results}/")

    ax.axhline(cfg.target_return, ls="--", color="grey", lw=1, label=f"target {cfg.target_return:g}")
    ax.set_xlabel("Online environment steps")
    ax.set_ylabel("Greedy evaluation return")
    ax.set_title(args.env)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(figures_dir(args.results), f"curve_{args.env}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
