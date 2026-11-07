"""Deep Q-Network agent (Mnih et al., 2015).

Two networks:
  * q         - the online network we train and act with
  * q_target  - a frozen copy used to compute learning targets, refreshed
                every `target_update` steps so the target does not move
                every time we take a gradient step

Loss for one transition (s, a, r, s', done):
    target = r + gamma * (1 - done) * max_a' Q_target(s', a')
    loss   = Huber(Q(s, a) - target)

Double DQN (van Hasselt et al., 2016) changes one thing: the ONLINE network
picks the best next action and the TARGET network scores it. Plain DQN lets
one network do both, which tends to overestimate Q-values.

CQL (Kumar et al., 2020) adds one term to the loss, used for OFFLINE learning:
    cql = logsumexp over actions of Q(s, a)  -  Q(s, a that the dataset took)
It pushes down the Q-values of actions the dataset gives no evidence for, so the
agent does not trust scores it has never seen proof of. cfg.cql_alpha is its
weight; 0.0 switches it off, which is plain DQN.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from rl.config import DQNConfig
from rl.networks import QNetwork


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: str = "cpu"):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device

        self.q = QNetwork(obs_dim, n_actions, cfg.hidden).to(device)
        self.q_target = copy.deepcopy(self.q)           # starts identical
        self.q_target.requires_grad_(False)             # never trained directly
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    # ------------------------------------------------------------------ acting
    def act(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: random action with probability epsilon, otherwise
        the action with the highest Q-value."""
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        return self.greedy_action(obs)

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(obs_t).argmax(dim=1).item())

    # ---------------------------------------------------------------- learning
    @torch.no_grad()
    def compute_targets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        next_obs = batch["next_obs"]
        if self.cfg.double_dqn:
            best_next = self.q(next_obs).argmax(dim=1, keepdim=True)       # online net chooses
            next_q = self.q_target(next_obs).gather(1, best_next).squeeze(1)  # target net scores
        else:
            next_q = self.q_target(next_obs).max(dim=1).values
        # (1 - terminated) removes the future term when the episode truly ended.
        return batch["rewards"] + self.cfg.gamma * (1.0 - batch["terminated"]) * next_q

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step on a minibatch. Returns numbers worth logging."""
        q_all = self.q(batch["obs"])                                        # Q(s, every action)
        q_taken = q_all.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)  # Q(s, a actually taken)
        targets = self.compute_targets(batch)

        td_loss = F.smooth_l1_loss(q_taken, targets)   # Huber loss, as in Mnih et al.

        # CQL penalty (Kumar et al., 2020, Eq. 4; exact for discrete actions).
        # logsumexp is a smooth "max" over ALL actions' scores in each state. Subtracting
        # q_taken (the score of the action the data actually took) means:
        #   - lowering the scores of every action pulls the penalty down, but
        #   - raising the score of the data's action also pulls it down.
        # Net effect: scores of actions the data never shows get pushed down.
        cql = (torch.logsumexp(q_all, dim=1) - q_taken).mean()
        loss = td_loss + self.cfg.cql_alpha * cql      # alpha = 0.0 leaves plain DQN

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item(), "td_loss": td_loss.item(), "cql": cql.item(),
                "q_mean": q_taken.mean().item()}

    def sync_target(self) -> None:
        """Hard update: copy the online network's weights into the target."""
        self.q_target.load_state_dict(self.q.state_dict())

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.sync_target()
