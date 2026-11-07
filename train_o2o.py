"""Configuration 3: OFFLINE-TO-ONLINE (the "naive" baseline).

Step 1 was done by train_offline.py: a Q-network learned from a fixed dataset with CQL.
Step 2 (this script): load those weights, then keep learning by PLAYING the game, with the
dataset still in the replay buffer next to the new online data. "Naive" = the agent just
explores epsilon-greedy. Task 3 replaces this collection rule with uncertainty-guided collection.

Needs the offline run first:
    python train_offline.py --env CartPole-v1 --policy medium --seed 0
Then:
    python train_o2o.py --env CartPole-v1 --policy medium --seed 0

Outputs (in results/o2o/<env>/<policy>/s<seed>/, see rl/paths.py):
    eval.csv      greedy return every `eval_every` ONLINE steps (row 0 = before any play)
    episodes.csv  return of every online training episode
    summary.json  steps to target (online steps only) and final return
    model.pt      final weights
"""
import argparse
import dataclasses
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
    parser.add_argument("--policy", choices=["random", "medium"], default="medium",
                        help="which dataset (and which offline-trained weights) to start from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="online step budget (default: total_steps, same as pure online)")
    parser.add_argument("--alpha", type=float, default=None, help="override the online CQL weight")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results", help="results root folder")
    args = parser.parse_args()

    cfg = get_config(args.env)
    alpha = cfg.o2o_alpha if args.alpha is None else args.alpha
    cfg = dataclasses.replace(cfg, cql_alpha=alpha)
    online_budget = args.steps or cfg.total_steps
    set_seed(args.seed)

    # The buffer must be big enough for dataset + everything we will collect. If it were only
    # as big as the dataset, the circular buffer would start overwriting the dataset from slot 0.
    data_path = f"{args.data}/{cfg.env_id}_{args.policy}_{cfg.dataset_size}"
    n_dataset = len(ReplayBuffer.load(data_path + ".npz"))
    buffer = ReplayBuffer.load(data_path + ".npz", capacity=n_dataset + online_budget)
    assert buffer.pos == n_dataset and buffer.capacity >= n_dataset + online_budget
    dataset_mean_return = json.load(open(data_path + ".json"))["mean_return"]

    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Start from the CQL-trained weights (load() also copies them into the target network).
    agent = DQNAgent(obs_dim, n_actions, cfg)
    offline_dir = run_dir("offline", cfg.env_id, args.seed, args.policy, root=args.out)
    agent.load(f"{offline_dir}/model.pt")

    tag = f"o2o-{args.policy}_{cfg.env_id}_s{args.seed}"
    out_dir = run_dir("o2o", cfg.env_id, args.seed, args.policy, root=args.out)
    os.makedirs(out_dir, exist_ok=True)
    eval_log = CSVLogger(f"{out_dir}/eval.csv",
                         ["env_steps", "episodes", "eval_return", "eval_std", "epsilon",
                          "loss", "td_loss", "cql", "q_mean"])
    ep_log = CSVLogger(f"{out_dir}/episodes.csv", ["env_steps", "episode", "return", "length"])

    # Convergence counts ONLY online steps (the dataset was free), so it compares fairly with pure online.
    steps_to_target, episodes_to_target = None, None
    eval_seed = 10_000 + 100 * args.seed
    nan = float("nan")
    last_stats = {"loss": nan, "td_loss": nan, "cql": nan, "q_mean": nan}

    # Row 0: how good is the offline agent before it plays a single online step?
    start_return, start_std = evaluate(agent, cfg.env_id, cfg.eval_episodes, eval_seed)
    eval_log.log({"env_steps": 0, "episodes": 0, "eval_return": start_return, "eval_std": start_std,
                  "epsilon": cfg.o2o_eps_start, **last_stats})
    if start_return >= cfg.target_return:
        steps_to_target, episodes_to_target = 0, 0
    print(f"[{tag}] start from offline weights: eval {start_return:.1f} ± {start_std:.1f} "
          f"| dataset {n_dataset:,} transitions (avg return {dataset_mean_return}) | online CQL alpha = {alpha:g}")

    obs, _ = env.reset(seed=args.seed)
    episode, ep_return, ep_len = 0, 0.0, 0
    start = time.time()

    for step in range(1, online_budget + 1):
        # 1. Act (epsilon-greedy, small epsilon) and store the new transition NEXT TO the dataset.
        epsilon = linear_schedule(cfg.o2o_eps_start, cfg.eps_end, cfg.eps_decay_steps, step)
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

        # 2. Learn from a random minibatch of dataset + online data mixed.
        #    No learning_starts wait: the buffer already holds the dataset.
        if step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                last_stats = agent.update(buffer.sample(cfg.batch_size))

        # 3. Refresh the target network (same schedule as pure online).
        if step % cfg.target_update == 0:
            agent.sync_target()

        # 4. Evaluate the greedy policy.
        if step % cfg.eval_every == 0:
            mean_ret, std_ret = evaluate(agent, cfg.env_id, cfg.eval_episodes, eval_seed)
            eval_log.log({"env_steps": step, "episodes": episode, "eval_return": mean_ret,
                          "eval_std": std_ret, "epsilon": round(epsilon, 3), **last_stats})
            if steps_to_target is None and mean_ret >= cfg.target_return:
                steps_to_target, episodes_to_target = step, episode
            print(f"[{tag}] step {step:>7} | episodes {episode:>4} | eval {mean_ret:8.1f} ± {std_ret:6.1f}"
                  f" | eps {epsilon:.2f} | loss {last_stats['loss']:.3f} | mean Q {last_stats['q_mean']:.1f}")

    eval_log.close()
    ep_log.close()
    agent.save(f"{out_dir}/model.pt")

    final_ret, final_std = evaluate(agent, cfg.env_id, 20, eval_seed + 50)
    summary = {
        "mode": f"o2o-{args.policy}", "env": cfg.env_id, "seed": args.seed,
        "dataset": data_path + ".npz", "dataset_mean_return": dataset_mean_return,
        "online_cql_alpha": alpha,
        "online_steps_used": online_budget,
        "target_return": cfg.target_return,
        "start_return": start_return,               # offline agent, before any online play
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
