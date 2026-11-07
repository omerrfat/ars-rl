"""One-off: move the old flat results/ files into the folder layout of rl/paths.py.

Prints every move. Refuses to overwrite (stops with an error if a destination exists).
Nothing is deleted and nothing is rerun.

    python tools/migrate_results.py            # does the moves
    python tools/migrate_results.py --dry-run  # only prints them
"""
import argparse
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.getcwd())
from rl.paths import run_dir  # noqa: E402

ROOT = "results"
# online_CartPole-v1_s0_eval.csv  /  offline-medium_Acrobot-v1_s2_summary.json  /  o2o-random_...
PATTERN = re.compile(r"^(online|offline|o2o)(?:-(random|medium))?_([A-Za-z]+-v\d)_s(\d+)_(.+)$")
NAMES = {"eval.csv", "episodes.csv", "summary.json", "model.pt"}


def move(src, dst, dry):
    if os.path.exists(dst):
        raise SystemExit(f"REFUSING to overwrite {dst} (from {src})")
    print(f"{src}  ->  {dst}")
    if not dry:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    dry = parser.parse_args().dry_run

    plans = []
    for path in sorted(glob.glob(f"{ROOT}/*")):
        name = os.path.basename(path)
        if os.path.isdir(path):
            continue
        if name.endswith(".png"):
            plans.append((path, os.path.join(ROOT, "figures", name)))
        elif name.endswith((".log", ".txt")):
            plans.append((path, os.path.join("logs", name)))
        else:
            m = PATTERN.match(name)
            if not m or m.group(5) not in NAMES:
                plans.append((path, os.path.join(ROOT, "archive", "unclear", name)))   # unknown: never guess
                continue
            mode, dataset, env, seed, fname = m.groups()
            plans.append((path, os.path.join(run_dir(mode, env, int(seed), dataset, ROOT), fname)))

    for src, dst in plans:
        move(src, dst, dry)
    print(f"{len(plans)} files {'would be ' if dry else ''}moved.")


if __name__ == "__main__":
    main()
