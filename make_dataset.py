"""Offline dataset generator.

Offline RL learns from a FIXED set of transitions that were collected
beforehand, with no game interaction while learning. This script makes those
datasets. A dataset is just a ReplayBuffer filled by some policy and saved.

Two kinds of dataset:
  random  actions chosen uniformly at random. Poor play, but it visits a wide
          range of states.
  medium  actions from a partly trained DQN (greedy, with a small chance of a
          random action). We stop its training the first time its skill
          reaches `medium_low` of the way from random play to the target
          return, and report how far along it really was.

Usage:
    python make_dataset.py --env CartPole-v1                 # both datasets
    python make_dataset.py --env Acrobot-v1 --policy random
    python make_dataset.py --env CartPole-v1 --policy medium --size 50000 --seed 1

Outputs (in data/):
    <env>_<policy>_<N>.npz    the transitions: obs, actions, rewards, next_obs, terminated
    <env>_<policy>_<N>.json   how the dataset was made and its average episode return
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np

from rl.config import get_config
from rl.dqn import DQNAgent
from rl.evaluate import evaluate
from rl.replay_buffer import ReplayBuffer
from rl.utils import linear_schedule, set_seed


def collect(cfg, n_steps, choose_action, seed):
    """Play the game for exactly `n_steps` steps, choosing actions with
    `choose_action(obs)`, and return (buffer, stats).

    The buffer is the dataset. Its capacity is exactly n_steps, so it never
    wraps around and overwrites anything.
    """
    env = gym.make(cfg.env_id)
    buffer = ReplayBuffer(env.observation_space.shape[0], n_steps)
    returns, n_terminated, n_truncated = [], 0, 0

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    for _ in range(n_steps):
        action = choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)   # same 5 fields as in training
        obs = next_obs
        ep_return += reward

        if terminated or truncated:
            returns.append(ep_return)
            n_terminated += int(terminated)                     # CartPole: pole fell. Acrobot: goal reached.
            n_truncated += int(truncated and not terminated)    # hit the 500-step time limit
            obs, _ = env.reset()
            ep_return = 0.0
    env.close()

    # Only finished games count towards the average. The last game may be cut
    # off when we reach n_steps; its transitions are kept but its score is not.
    stats = {
        "n_transitions": len(buffer),
        "n_episodes": len(returns),
        "mean_return": float(np.mean(returns)) if returns else None,
        "std_return": float(np.std(returns)) if returns else None,
        "n_terminated": n_terminated,
        "n_truncated": n_truncated,
    }
    return buffer, stats


def train_medium_agent(cfg, seed, low):
    """Train an ordinary online DQN (the same loop as train_online.py) but STOP
    as soon as its greedy score reaches `low`: the earliest moment it is clearly
    better than random play.

    Why stop at the first check above `low` instead of waiting for a score
    inside a band? Learning can jump: on Acrobot the score went from -500 to
    -186 within 100 steps, skipping any band in between. The first check above
    `low` is the least-trained agent the training run can give us.

    Returns (agent, step_it_stopped_at, its_greedy_score).
    """
    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(obs_dim, env.action_space.n, cfg)
    memory = ReplayBuffer(obs_dim, cfg.buffer_size)   # this agent's own memory, NOT the dataset
    eval_seed = 10_000 + 100 * seed

    obs, _ = env.reset(seed=seed)
    for step in range(1, cfg.total_steps + 1):
        # act, store, learn, sync: identical to train_online.py
        epsilon = linear_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, step)
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        memory.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                agent.update(memory.sample(cfg.batch_size))
        if step % cfg.target_update == 0:
            agent.sync_target()

        # Instead of "has it reached the target?", ask "has it reached `low` yet?".
        # We test often (every medium_check_every steps) so we stop soon after it gets there.
        if step >= cfg.learning_starts and step % cfg.medium_check_every == 0:
            score, _ = evaluate(agent, cfg.env_id, cfg.eval_episodes, eval_seed)
            print(f"[medium agent] step {step:>6} | greedy score {score:8.1f} | stop at {low:.1f} or better")
            if score >= low:
                return agent, step, score
    raise RuntimeError("Agent never reached the medium score within total_steps.")


def save_dataset(cfg, policy, buffer, stats, seed, out_dir, extra=None):
    """Write the transitions (.npz) and a small description (.json) next to them."""
    name = f"{cfg.env_id}_{policy}_{stats['n_transitions']}"
    npz_path = f"{out_dir}/{name}.npz"
    buffer.save(npz_path)

    info = {"env": cfg.env_id, "policy": policy, "seed": seed, **stats, **(extra or {})}
    with open(f"{out_dir}/{name}.json", "w") as f:
        json.dump(info, f, indent=2)

    # Read it back as a sanity check that save/load works. Offline training will rely on it.
    assert len(ReplayBuffer.load(npz_path)) == stats["n_transitions"]

    print(f"\nSaved {npz_path}")
    print(f"  transitions: {stats['n_transitions']:,} | finished episodes: {stats['n_episodes']}")
    print(f"  average episode return: {stats['mean_return']:.1f} +/- {stats['std_return']:.1f}")
    print(f"  episodes ended by termination: {stats['n_terminated']} | by time limit: {stats['n_truncated']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--policy", choices=["random", "medium", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=None, help="override dataset_size (transitions)")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    cfg = get_config(args.env)
    n = args.size or cfg.dataset_size
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    env = gym.make(cfg.env_id)
    n_actions = env.action_space.n
    env.close()

    def random_action(obs):
        return np.random.randint(n_actions)

    if args.policy in ("random", "both"):
        buffer, stats = collect(cfg, n, random_action, args.seed)
        save_dataset(cfg, "random", buffer, stats, args.seed, args.out, extra={"epsilon": 1.0})

    if args.policy in ("medium", "both"):
        # How good is random play here? We need this to turn "30% to 60% of the
        # way to the target" into actual scores for this environment.
        _, base = collect(cfg, 5_000, random_action, args.seed)
        random_score = base["mean_return"]
        low = random_score + cfg.medium_low * (cfg.target_return - random_score)
        high = random_score + cfg.medium_high * (cfg.target_return - random_score)
        print(f"Random play scores {random_score:.1f}; target is {cfg.target_return:g}; "
              f"medium agent: stop training at {low:.1f} or better (hoping to stay under {high:.1f})")

        agent, trained_steps, greedy_score = train_medium_agent(cfg, args.seed, low)
        progress = (greedy_score - random_score) / (cfg.target_return - random_score)
        print(f"Medium agent ready: {greedy_score:.1f} greedy, i.e. {progress:.0%} of the way from random play to the target")
        if greedy_score > high:
            print(f"  NOTE: above the intended band (top {cfg.medium_high:.0%}). The agent improved faster than "
                  f"the checks could catch. The dataset is still valid, just better than 'medium' ideally means.")

        # Collect with the medium agent: mostly its own choice, sometimes random.
        buffer, stats = collect(cfg, n, lambda obs: agent.act(obs, cfg.medium_epsilon), args.seed)
        # How good is the DATA, on the same 0-to-1 scale? (It can differ from the agent's own
        # test score: different games, and the small epsilon changes how the agent plays.)
        data_progress = (stats["mean_return"] - random_score) / (cfg.target_return - random_score)
        extra = {"epsilon": cfg.medium_epsilon, "agent_trained_steps": trained_steps,
                 "agent_greedy_score": greedy_score, "agent_progress": progress,
                 "dataset_progress": data_progress,
                 "random_baseline": random_score, "stop_score": low, "intended_band_top": high}
        save_dataset(cfg, "medium", buffer, stats, args.seed, args.out, extra=extra)
        print(f"  agent test score {greedy_score:.1f} ({progress:.0%} of the way) after {trained_steps:,} training steps; "
              f"dataset average {stats['mean_return']:.1f} ({data_progress:.0%} of the way)")


if __name__ == "__main__":
    main()
