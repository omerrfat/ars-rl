"""Replay buffer: the agent's memory of past transitions.

This one class serves all three configurations:
  * online:            starts empty and fills as the agent plays
  * offline:           loaded from a saved dataset file and never grows
  * offline-to-online: loaded from the dataset, then new online data is added

Each transition is (state, action, reward, next_state, terminated).
We store `terminated` (the episode really ended, e.g. the pole fell) and NOT
`truncated` (the 500-step time limit). At a time limit the future still has
value, so the Q-target must keep bootstrapping from the next state.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0        # where the next transition is written
        self.size = 0       # how many slots are filled

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        # Circular buffer: once full, overwrite the oldest transition.
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Uniform random minibatch. Random sampling breaks the correlation
        between consecutive steps, which is what makes DQN stable."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "terminated": torch.as_tensor(self.terminated[idx], device=device),
        }

    # ----- Saving and loading: this is what makes offline learning possible -----

    def save(self, path: str) -> None:
        n = self.size
        np.savez_compressed(
            path,
            obs=self.obs[:n], actions=self.actions[:n], rewards=self.rewards[:n],
            next_obs=self.next_obs[:n], terminated=self.terminated[:n],
        )

    @classmethod
    def load(cls, path: str, capacity: int | None = None) -> "ReplayBuffer":
        """Load a saved dataset. `capacity` can be larger than the dataset so
        that offline-to-online runs have room to add online transitions."""
        data = np.load(path)
        n = len(data["actions"])
        buf = cls(obs_dim=data["obs"].shape[1], capacity=max(capacity or n, n))
        for key in ("obs", "actions", "rewards", "next_obs", "terminated"):
            getattr(buf, key)[:n] = data[key]
        buf.size = n
        buf.pos = n % buf.capacity
        return buf
