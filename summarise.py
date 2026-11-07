"""Summarise ALL three configurations: tables, figures and results/RESULTS.md.

Usage:
    python summarise.py

Reads every finished run under results/ (see rl/paths.py) and writes
    results/RESULTS.md                    overview, regenerated every time (never edit by hand)
    results/figures/curves_<env>.png      online vs O2O-random vs O2O-medium, x = online env steps
    results/figures/offline_curves.png    offline CQL curves, x = gradient updates (summarise_offline.py)
"""
import contextlib
import datetime
import io
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import summarise_offline
from rl.config import get_config
from rl.paths import figures_dir, find_runs

ROOT = "results"
ENVS = ["CartPole-v1", "Acrobot-v1"]
# (row label, mode, dataset)
CONFIGS = [
    ("online", "online", None),
    ("offline CQL, random data", "offline", "random"),
    ("offline CQL, medium data", "offline", "medium"),
    ("O2O, random data", "o2o", "random"),
    ("O2O, medium data", "o2o", "medium"),
]
DIP_WINDOW = 6_000          # "fine-tuning dip": lowest eval within the first N online steps
LAST_N = 5                  # evals averaged for the "last 5 evals" column


# ------------------------------------------------------------------ loading
def load(mode, env, dataset):
    """List of dicts, one per finished run: seed, summary (json) and eval (DataFrame)."""
    runs = []
    for seed, folder in find_runs(mode, env, dataset, ROOT):
        runs.append({"seed": seed, "summary": json.load(open(os.path.join(folder, "summary.json"))),
                     "eval": pd.read_csv(os.path.join(folder, "eval.csv"))})
    return runs


def online_evals(run, mode):
    """The eval rows that count as ONLINE play. O2O row 0 is the offline agent, before any play."""
    df = run["eval"]
    return df[df["env_steps"] > 0] if mode == "o2o" else df


# --------------------------------------------------------------- formatting
def mean_sd(values, digits=1):
    """'mean ± sample SD' (ddof=1: a few seeds are a sample). One value gives just the mean."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def reached(values, n_seeds, digits=0):
    """Mean ± SD over only the seeds that reached the target, with 'k/n' if not all did."""
    got = [x for x in values if x is not None]
    if not got:
        return f"not reached (0/{n_seeds})"
    text = mean_sd(got, digits)
    return text if len(got) == n_seeds else f"{text} ({len(got)}/{n_seeds} reached)"


# ------------------------------------------------------------------- tables
def table_rows(env):
    rows = []
    for label, mode, dataset in CONFIGS:
        runs = load(mode, env, dataset)
        if not runs:
            continue
        n = len(runs)
        sums = [r["summary"] for r in runs]
        if mode == "offline":
            used, to_target, episodes = "0", "n/a (gradient updates)", "n/a"
            evals = [r["eval"] for r in runs]
        else:
            used = mean_sd([s["online_steps_used"] for s in sums], 0)
            to_target = reached([s["steps_to_target"] for s in sums], n)
            episodes = reached([s["episodes_to_target"] for s in sums], n)
            evals = [online_evals(r, mode) for r in runs]
        rows.append([
            label, used, to_target, episodes,
            mean_sd([s["final_return"] for s in sums]),
            mean_sd([df["eval_return"].tail(LAST_N).mean() for df in evals]),
            mean_sd([df["eval_return"].max() for df in evals]),
            mean_sd([df["q_mean"].iloc[-1] for df in evals]),
            str(n),
        ])
    return rows


HEADER = ["Configuration", "Online steps used", "Steps to target", "Episodes to target",
          "Final return (20 eps)", f"Mean of last {LAST_N} evals", "Best eval", "Last mean Q", "Seeds"]


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def dip_rows(env):
    """Per O2O run: eval at online step 0 vs the lowest eval in the first DIP_WINDOW online steps."""
    rows = []
    for label, mode, dataset in CONFIGS:
        if mode != "o2o":
            continue
        for r in load(mode, env, dataset):
            df = r["eval"]
            start = df.loc[df["env_steps"] == 0, "eval_return"].iloc[0]
            early = df[(df["env_steps"] > 0) & (df["env_steps"] <= DIP_WINDOW)]
            low = early.loc[early["eval_return"].idxmin()]
            change = low["eval_return"] - start
            rows.append([label, f"s{r['seed']}", f"{start:.1f}", f"{low['eval_return']:.1f}",
                         f"{int(low['env_steps']):,}", f"{change:+.1f}" if change < 0 else "no dip"])
    return rows


DIP_HEADER = ["Configuration", "Seed", "Eval at step 0", f"Lowest in first {DIP_WINDOW:,} steps",
              "At step", "Change"]


# ------------------------------------------------------------------ figures
def curves_figure(env):
    cfg = get_config(env)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, mode, dataset in [c for c in CONFIGS if c[1] != "offline"]:
        runs = load(mode, env, dataset)
        if not runs:
            continue
        n = min(len(r["eval"]) for r in runs)                 # align runs to the shortest
        steps = runs[0]["eval"]["env_steps"].to_numpy()[:n]
        ret = np.stack([r["eval"]["eval_return"].to_numpy()[:n] for r in runs])
        line, = ax.plot(steps, ret.mean(axis=0), label=f"{label} (mean of {len(runs)} seeds)")
        ax.fill_between(steps, ret.min(axis=0), ret.max(axis=0), alpha=0.15, color=line.get_color())
        if mode == "o2o":
            # O2O row 0 IS the offline CQL agent's score: mark it as the starting point at x = 0.
            start = ret[:, 0].mean()
            ax.scatter([0], [start], marker="D", s=60, zorder=5, color=line.get_color(),
                       edgecolor="black", label=f"offline CQL {dataset} start ({start:.0f})")
    ax.axhline(cfg.target_return, ls="--", color="grey", lw=1, label=f"target {cfg.target_return:g}")
    ax.set_xlabel("Online environment steps")
    ax.set_ylabel("Greedy evaluation return")
    ax.set_title(f"{env}: mean line, min-max band over seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(figures_dir(ROOT), f"curves_{env}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------- RESULTS.md
HOW_TO_READ = f"""## How to read this

- **Return** = total reward in one game. Higher is better in both environments.
  CartPole: number of steps the pole stays up (max 500). Acrobot: minus the number of steps taken
  (so closer to 0 is better; -500 means it never reached the goal).
- **Targets** (used for "steps to target"): CartPole **475**, Acrobot **-100**.
- **Q sanity limits** (mean Q of the actions taken): CartPole must be **<= 100** (reward +1 per step,
  gamma 0.99, so 1/(1-0.99) = 100); Acrobot must be **<= 0** (every step costs -1). A mean Q above its limit
  means the network is overestimating.
- **Which number to trust most:** the *mean of the last {LAST_N} evals* (smooths single lucky or unlucky tests),
  then the *final return* (20 episodes at the very end). *Best eval* is optimistic: it is the highest of many
  noisy tests, and so is "steps to target" (the first test that crossed the line may be a spike).
- **Steps to target** counts ONLINE environment steps only. Offline runs use no online steps; their targets
  are counted in gradient updates, which are not comparable with game steps.
- **± values** are the sample SD across seeds. "(k/n reached)" means only k of n seeds reached the target, and
  the mean is over those k.
- **O2O** = offline-to-online: start from the offline CQL weights and keep learning by playing.
  Its eval at online step 0 is the offline agent before any play."""

FOLDERS = """## Folders

- `results/online/<env>/s<seed>/` : pure online DQN run (eval.csv, episodes.csv, summary.json, model.pt)
- `results/offline/<env>/<dataset>/s<seed>/` : offline CQL run from the random or medium dataset (eval.csv, summary.json, model.pt)
- `results/o2o/<env>/<dataset>/s<seed>/` : offline-to-online run, started from the matching offline seed (eval.csv, episodes.csv, summary.json, model.pt)
- `results/figures/` : every .png (generated by summarise.py, plot.py)
- `results/archive/` : old or superseded runs, kept but not used in any table
- `logs/` : console output of the batch runs
- `data/` : the offline datasets (.npz + .json with the dataset's average return)"""


def settings_text():
    lines = []
    for env in ENVS:
        c = get_config(env)
        seeds = {label: [s for s, _ in find_runs(mode, env, ds, ROOT)] for label, mode, ds in CONFIGS}
        seeds_txt = "; ".join(f"{k}: {v}" for k, v in seeds.items() if v)
        lines.append(
            f"- **{env}**: online budget {c.total_steps:,} steps, eval every {c.eval_every:,} online steps "
            f"({c.eval_episodes} greedy episodes; {c.offline_eval_every:,} updates offline), "
            f"offline {c.offline_steps:,} updates at CQL alpha {c.offline_alpha:g}, "
            f"dataset {c.dataset_size:,} transitions, O2O epsilon {c.o2o_eps_start:g} -> {c.eps_end:g} over "
            f"{c.eps_decay_steps:,} steps with online CQL alpha {c.o2o_alpha:g}. Seeds found: {seeds_txt}.")
    lines.append("- Same dataset (seed 0) for every training seed. Eval every N steps is coarse "
                 "(a spike between two evals is invisible). Results reproduce exactly on the same machine, but may differ on another machine.")
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")        # so ± prints on the Windows console

    parts = ["# Results overview",
             f"*Generated by `python summarise.py` on {datetime.date.today().isoformat()}. "
             "Do not edit by hand; rerun the script.*", "", HOW_TO_READ, ""]
    for env in ENVS:
        cfg = get_config(env)
        path = curves_figure(env)
        parts += [f"## {env} (target {cfg.target_return:g})", "",
                  markdown_table(HEADER, table_rows(env)), "",
                  f"### Fine-tuning dip (O2O): eval at online step 0 vs the lowest in the first {DIP_WINDOW:,} steps", "",
                  markdown_table(DIP_HEADER, dip_rows(env)), "",
                  f"![{env} learning curves](figures/{os.path.basename(path)})", ""]
    with contextlib.redirect_stdout(io.StringIO()):           # offline figure; its own table is not needed here
        summarise_offline.main()
    parts += ["## Offline CQL curves (x = gradient updates, one line per seed)", "",
              "![offline curves](figures/offline_curves.png)", "",
              "## Settings and seeds used", "", settings_text(), "", FOLDERS, ""]

    text = "\n".join(parts)
    with open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nWrote {ROOT}/RESULTS.md and figures in {ROOT}/figures/")


if __name__ == "__main__":
    main()
