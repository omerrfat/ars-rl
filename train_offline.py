"""Configuration 2: pure OFFLINE learning with DQN + CQL.

The agent never plays the game while it learns. It is handed a fixed dataset
(made by make_dataset.py) and learns only from that. The game is used for one
thing only: TESTING the agent every `offline_eval_every` gradient updates.

Usage:
    python train_offline.py --env CartPole-v1 --policy medium --seed 0
    python train_offline.py --env Acrobot-v1 --policy random --seed 0
    python train_offline.py --env CartPole-v1 --policy medium --alpha 0   # alpha 0 = no CQL

Outputs (in results/offline/<env>/<policy>/s<seed>/, see rl/paths.py):
    eval.csv      greedy return every `offline_eval_every` updates
    summary.json  convergence (in gradient steps) and final return
    model.pt      trained Q-network weights (train_o2o.py starts from these)
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
from rl.utils import CSVLogger, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--policy", choices=["random", "medium"], default="medium",
                        help="which dataset to learn from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=None, help="dataset size (default: dataset_size)")
    parser.add_argument("--steps", type=int, default=None, help="override offline_steps (gradient updates)")
    parser.add_argument("--alpha", type=float, default=None, help="override the CQL weight (0 = plain DQN)")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results", help="results root folder")
    args = parser.parse_args()

    cfg = get_config(args.env)
    # Same settings as online, except the CQL weight. `replace` makes a modified copy of the config.
    alpha = cfg.offline_alpha if args.alpha is None else args.alpha
    cfg = dataclasses.replace(cfg, cql_alpha=alpha)
    n_updates = args.steps or cfg.offline_steps
    set_seed(args.seed)

    # The dataset IS the replay buffer, loaded from the file and never added to.
    # No env.step() happens anywhere below except inside evaluate().
    data_path = f"{args.data}/{cfg.env_id}_{args.policy}_{args.size or cfg.dataset_size}"
    buffer = ReplayBuffer.load(data_path + ".npz")
    dataset_mean_return = None
    if os.path.exists(data_path + ".json"):
        dataset_mean_return = json.load(open(data_path + ".json"))["mean_return"]

    obs_dim = buffer.obs.shape[1]
    env = gym.make(cfg.env_id)          # only to ask how many actions exist
    n_actions = env.action_space.n
    env.close()
    agent = DQNAgent(obs_dim, n_actions, cfg)

    tag = f"offline-{args.policy}_{cfg.env_id}_s{args.seed}"
    out_dir = run_dir("offline", cfg.env_id, args.seed, args.policy, root=args.out)
    os.makedirs(out_dir, exist_ok=True)
    eval_log = CSVLogger(f"{out_dir}/eval.csv",
                         ["gradient_steps", "eval_return", "eval_std", "loss", "td_loss", "cql", "q_mean"])

    # Convergence speed, counted in gradient updates (there are no game steps to count).
    steps_to_target = None
    best_eval = None
    eval_seed = 10_000 + 100 * args.seed     # same test start states as the online runs
    last_stats = {"loss": float("nan"), "td_loss": float("nan"), "cql": float("nan"), "q_mean": float("nan")}
    start = time.time()

    print(f"[{tag}] learning from {len(buffer):,} transitions (dataset average return "
          f"{dataset_mean_return}), CQL alpha = {alpha:g}")

    for step in range(1, n_updates + 1):
        # 1. Learn from a random minibatch of the FIXED dataset.
        last_stats = agent.update(buffer.sample(cfg.batch_size))

        # 2. Refresh the target network (counted in updates, since there are no game steps).
        if step % cfg.offline_target_update == 0:
            agent.sync_target()

        # 3. Test the greedy policy in the real game.
        if step % cfg.offline_eval_every == 0:
            mean_ret, std_ret = evaluate(agent, cfg.env_id, cfg.eval_episodes, eval_seed)
            eval_log.log({"gradient_steps": step, "eval_return": mean_ret, "eval_std": std_ret, **last_stats})
            best_eval = mean_ret if best_eval is None else max(best_eval, mean_ret)
            if steps_to_target is None and mean_ret >= cfg.target_return:
                steps_to_target = step
            print(f"[{tag}] update {step:>6} | eval {mean_ret:8.1f} ± {std_ret:6.1f}"
                  f" | loss {last_stats['loss']:.3f} | cql {last_stats['cql']:.3f} | mean Q {last_stats['q_mean']:.1f}")

    eval_log.close()
    agent.save(f"{out_dir}/model.pt")

    final_ret, final_std = evaluate(agent, cfg.env_id, 20, eval_seed + 50)
    summary = {
        "mode": f"offline-{args.policy}", "env": cfg.env_id, "seed": args.seed,
        "dataset": data_path + ".npz", "dataset_mean_return": dataset_mean_return,
        "cql_alpha": alpha,
        "online_steps_used": 0,                 # offline: the agent never plays while learning
        "gradient_steps_used": n_updates,
        "target_return": cfg.target_return,
        "gradient_steps_to_target": steps_to_target,
        "best_eval": best_eval,
        "final_return": final_ret, "final_std": final_std,
        "wall_time_s": round(time.time() - start, 1),
    }
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
