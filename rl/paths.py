"""Where every run's files live. All scripts get their paths from here, so the folder
layout is defined in exactly ONE place.

    results/online/<env>/s<seed>/              eval.csv, episodes.csv, summary.json, model.pt
    results/offline/<env>/<dataset>/s<seed>/   eval.csv, summary.json, model.pt
    results/o2o/<env>/<dataset>/s<seed>/       eval.csv, episodes.csv, summary.json, model.pt
    results/figures/                           every .png
    results/archive/                           old or superseded runs
    logs/                                      console output (.log / .txt)

<dataset> is "random" or "medium". Online runs have no dataset.
"""
import glob
import os

ROOT = "results"
MODES = ("online", "offline", "o2o")


def run_dir(mode: str, env: str, seed: int, dataset: str | None = None, root: str = ROOT) -> str:
    """Folder for ONE run (not created here; use os.makedirs when writing)."""
    assert mode in MODES, f"mode must be one of {MODES}"
    if mode == "online":
        return os.path.join(root, "online", env, f"s{seed}")
    assert dataset in ("random", "medium"), "offline and o2o runs need dataset='random' or 'medium'"
    return os.path.join(root, mode, env, dataset, f"s{seed}")


def find_runs(mode: str, env: str, dataset: str | None = None, root: str = ROOT) -> list[tuple[int, str]]:
    """Every existing run folder of one kind as (seed, folder), sorted by seed.
    A run counts as existing once its summary.json was written (i.e. it finished)."""
    base = os.path.join(root, mode, env) if mode == "online" else os.path.join(root, mode, env, dataset)
    runs = []
    for summary in glob.glob(os.path.join(base, "s*", "summary.json")):
        folder = os.path.dirname(summary)
        runs.append((int(os.path.basename(folder)[1:]), folder))
    return sorted(runs)


def figures_dir(root: str = ROOT) -> str:
    path = os.path.join(root, "figures")
    os.makedirs(path, exist_ok=True)
    return path
