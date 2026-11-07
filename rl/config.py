"""Hyperparameters for every environment, kept in one place.

Each experiment reads its settings from here, so all conditions (online,
offline, offline-to-online) share the same network size, learning rate and
evaluation protocol. Only the way the agent gets its data changes.
"""
from dataclasses import dataclass


@dataclass
class DQNConfig:
    env_id: str
    total_steps: int          # online environment steps for the pure online run
    target_return: float      # "solved" score used for convergence speed

    # Learning
    lr: float = 5e-4          # Adam learning rate
    gamma: float = 0.99       # discount factor: how much future reward counts
    batch_size: int = 128     # transitions sampled per gradient update
    hidden: int = 128         # units in each of the two hidden layers
    max_grad_norm: float = 10.0  # gradient clipping, stops rare huge updates
    double_dqn: bool = True   # use Double DQN targets (see dqn.py)

    # Replay buffer and target network (the two DQN stabilisers)
    buffer_size: int = 100_000
    learning_starts: int = 1_000   # collect this many steps before learning
    train_freq: int = 1            # run a training burst every N env steps
    gradient_steps: int = 1        # gradient updates per training burst
    target_update: int = 500       # copy online net -> target net every N env steps

    # Exploration: epsilon-greedy, decayed linearly
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000

    # Evaluation
    eval_every: int = 2_000        # evaluate the greedy policy every N steps
    eval_episodes: int = 10        # episodes per evaluation

    # Offline dataset generation (make_dataset.py)
    dataset_size: int = 20_000     # transitions saved in each offline dataset
    # A "medium" agent is a partly trained DQN. We measure "how trained" as the
    # fraction of the way its greedy score is from random play (0.0) to the
    # target return (1.0). This works for CartPole (higher is better) and for
    # Acrobot (negative scores) alike.
    medium_low: float = 0.3        # stop training the first time the agent is at least this far along
    medium_high: float = 0.6       # intended upper edge of "medium". Only a warning if exceeded, because
                                   # learning can jump past it between two checks (it does on Acrobot)
    medium_check_every: int = 100  # test the agent this often while training it
    medium_epsilon: float = 0.05   # chance of a random action while collecting with the medium agent

    # CQL and offline training (dqn.py, train_offline.py)
    cql_alpha: float = 0.0         # weight of the CQL penalty in the loss. 0.0 = plain DQN (every online run)
    offline_alpha: float = 1.0     # the cql_alpha train_offline.py uses. Kumar et al. say to tune it; we start at 1.0
    offline_steps: int = 20_000    # gradient updates in an offline run (no game steps happen at all)
    offline_eval_every: int = 1_000    # test the greedy policy every N gradient updates
    offline_target_update: int = 250   # copy online net -> target net every N gradient updates

    # Offline-to-online (train_o2o.py)
    o2o_alpha: float = 0.0         # CQL weight while playing online. 0.0 = plain DQN; the new online data is
                                   # real, so there is less reason to distrust scores for unseen actions
    o2o_eps_start: float = 0.1     # start exploring a little, not 1.0: the agent already knows something.
                                   # Then decays linearly to eps_end over eps_decay_steps, as in online


CONFIGS = {
    # CartPole: 2 actions, +1 reward per step, max 500 steps. "Solved" = 475.
    # Settings follow the tuned DQN recipe in RL Baselines3 Zoo (Raffin, 2020):
    # train in bursts (128 updates every 256 steps) and refresh the target
    # network before each burst. Updating every single step plateaued
    # around a return of 100-200 in our tests.
    "CartPole-v1": DQNConfig(
        env_id="CartPole-v1",
        total_steps=50_000,
        target_return=475.0,
        lr=2.3e-3,
        batch_size=64,
        hidden=256,
        train_freq=256,
        gradient_steps=128,
        target_update=10,
        eps_end=0.04,
        eps_decay_steps=8_000,
        offline_target_update=128,   # online, the target stays frozen for one burst of 128 updates
    ),
    # Acrobot: 3 actions, -1 reward per step until the tip swings above the
    # line, max 500 steps. Gymnasium's "solved" level is -100.
    # Starting point taken from the same Zoo recipe; not yet tuned here.
    "Acrobot-v1": DQNConfig(
        env_id="Acrobot-v1",
        total_steps=100_000,
        target_return=-100.0,
        lr=6.3e-4,
        batch_size=128,
        hidden=256,
        buffer_size=50_000,
        train_freq=4,
        gradient_steps=4,
        target_update=250,
        eps_end=0.1,
        eps_decay_steps=12_000,
    ),
}


def get_config(env_id: str) -> DQNConfig:
    if env_id not in CONFIGS:
        raise ValueError(f"No config for {env_id}. Options: {list(CONFIGS)}")
    return CONFIGS[env_id]
