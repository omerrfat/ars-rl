"""Configuration 1: pure ONLINE learning with DQN.

The agent starts with an empty replay buffer and learns only from the data
it collects itself.

Usage:
    python train_online.py --env CartPole-v1 --seed 0
    python train_online.py --env Acrobot-v1 --seed 0 --save-buffer

Outputs (in results/online/<env>/s<seed>/, see rl/paths.py):
    eval.csv      greedy return every `eval_every` steps
    episodes.csv  return of every training episode
    summary.json  convergence speed and final return
    model.pt      trained Q-network weights
"""
import argparse
import json
import os
import time

import gymnasium as gym

from rl.config import get_config
from rl.dqn import DQNAgent
from rl.evaluate import evaluate
from rl.paths import run_dir
from rl.replay_buffer import ReplayBuffer
from rl.utils import CSVLogger, linear_schedule, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="override total_steps")
    parser.add_argument("--out", default="results", help="results root folder")
    parser.add_argument("--save-buffer", action="store_true",
                        help="also save the replay buffer, usable later as an offline dataset")
    args = parser.parse_args()

    cfg = get_config(args.env)  # look up the game settings.
    total_steps = args.steps or cfg.total_steps
    set_seed(args.seed)

    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, cfg)
    buffer = ReplayBuffer(obs_dim, cfg.buffer_size)

    tag = f"online_{cfg.env_id}_s{args.seed}"
    out_dir = run_dir("online", cfg.env_id, args.seed, root=args.out)
    os.makedirs(out_dir, exist_ok=True)
    eval_log = CSVLogger(f"{out_dir}/eval.csv",
                         ["env_steps", "episodes", "eval_return", "eval_std", "epsilon",
                          "loss", "td_loss", "cql", "q_mean"])
    ep_log = CSVLogger(f"{out_dir}/episodes.csv", ["env_steps", "episode", "return", "length"])

    # Convergence speed: the first evaluation where the greedy policy reaches
    # the target return. We record it in both env steps and episodes.
    steps_to_target, episodes_to_target = None, None
    eval_seed = 10_000 + 100 * args.seed   # evaluation start states differ from training ones

    obs, _ = env.reset(seed=args.seed)
    episode, ep_return, ep_len = 0, 0.0, 0
    last_stats = {"loss": float("nan"), "td_loss": float("nan"), "cql": float("nan"),
                  "q_mean": float("nan")}
    start = time.time()

    for step in range(1, total_steps + 1):
        # 1. Act (epsilon-greedy) and store the transition.
        epsilon = linear_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, step)
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if terminated or truncated:
            episode += 1
            ep_log.log({"env_steps": step, "episode": episode, "return": ep_return, "length": ep_len})
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # 2. Learn from a random minibatch of past experience.
        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                last_stats = agent.update(buffer.sample(cfg.batch_size))

        # 3. Refresh the target network.
        if step % cfg.target_update == 0: # every 10 steps in CartPole, copies the student into the frozen copy.
            agent.sync_target()

        # 4. Evaluate the greedy policy.
        if step % cfg.eval_every == 0:
            mean_ret, std_ret = evaluate(agent, cfg.env_id, cfg.eval_episodes, eval_seed)
            eval_log.log({"env_steps": step, "episodes": episode, "eval_return": mean_ret,
                          "eval_std": std_ret, "epsilon": round(epsilon, 3), **last_stats})
            if steps_to_target is None and mean_ret >= cfg.target_return:
                steps_to_target, episodes_to_target = step, episode
            print(f"[{tag}] step {step:>7} | episodes {episode:>4} | eval {mean_ret:8.1f} ± {std_ret:6.1f}"
                  f" | eps {epsilon:.2f} | loss {last_stats['loss']:.3f}")

    eval_log.close()
    ep_log.close()
    agent.save(f"{out_dir}/model.pt")
    if args.save_buffer:
        os.makedirs("data", exist_ok=True)
        buffer.save(f"data/{tag}_buffer.npz")

    final_ret, final_std = evaluate(agent, cfg.env_id, 20, eval_seed + 50)
    summary = {
        "mode": "online", "env": cfg.env_id, "seed": args.seed,
        "online_steps_used": total_steps,
        "target_return": cfg.target_return,
        "steps_to_target": steps_to_target,
        "episodes_to_target": episodes_to_target,
        "final_return": final_ret, "final_std": final_std,
        "wall_time_s": round(time.time() - start, 1),
    }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
