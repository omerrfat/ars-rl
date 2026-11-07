"""Small helpers shared by every script."""
import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix every random number generator so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small networks run fastest on one CPU thread, and it lets several
    # seeds run side by side without fighting over cores.
    torch.set_num_threads(1)


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    """Epsilon decays in a straight line from `start` to `end` over `duration`
    steps, then stays at `end`."""
    fraction = min(step / duration, 1.0)
    return start + fraction * (end - start)


class CSVLogger:
    """Appends one row (a dict) per call to a CSV file."""

    def __init__(self, path: str, fieldnames: list[str]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: dict) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()
