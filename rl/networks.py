"""The Q-network: maps a state to one score per action.

Mnih et al. (2015) used a convolutional network because their input was
Atari pixels. Our states are short vectors (4 numbers for CartPole, 6 for
Acrobot), so a small fully connected network is enough.
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),  # one output = Q(s, a) for each action a
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
