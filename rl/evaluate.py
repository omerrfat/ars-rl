"""Evaluation: how good is the current policy, with no exploration?

We run the GREEDY policy (epsilon = 0) on a separate environment for a fixed
number of episodes. Using a separate environment keeps evaluation from
disturbing the training episode in progress.
"""
import gymnasium as gym
import numpy as np


def evaluate(agent, env_id: str, n_episodes: int, seed: int) -> tuple[float, float]:
    env = gym.make(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)   # fixed seeds: same start states every evaluation
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.greedy_action(obs))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))
