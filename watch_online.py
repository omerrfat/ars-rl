"""Watch online DQN learn CartPole, live, in a pygame window.

This is the SAME training loop as train_online.py (same agent, buffer, config
and evaluation). The only addition is drawing: the game, a metrics panel and
a learning-curve chart. It writes no files, so it never touches results/.

Usage:
    python watch_online.py                  # seed 0, medium speed
    python watch_online.py --seed 1 --speed 1

Keys while it runs:
    1 / 2 / 3   speed: 1 = slow (real time), 2 = medium, 3 = fast
    SPACE       pause / resume
    Q or ESC    quit

Because it uses the same seed handling and call order as train_online.py,
the same seed gives the same evaluation scores as that script (checked for
the first 6,000 steps of seed 0).
"""
import argparse
import time
from collections import deque

import gymnasium as gym
import numpy as np
import pygame

from rl.config import get_config
from rl.dqn import DQNAgent
from rl.evaluate import evaluate
from rl.replay_buffer import ReplayBuffer
from rl.utils import linear_schedule, set_seed

ENV_ID = "CartPole-v1"
MAX_RETURN = 500          # CartPole's best possible score; sets the chart's y-axis

# Speed settings: (name, draw once every N env steps, max draws per second; 0 = no limit).
# Drawing is slow compared with learning, so drawing less often = training faster.
SPEEDS = {
    1: ("1 slow (real time)", 1, 50),
    2: ("2 medium", 5, 0),
    3: ("3 fast", 100, 0),
}

# Window layout (pixels). The game frame from Gymnasium is 600 x 400.
WIN_W, WIN_H = 960, 640
GAME_W, GAME_H = 600, 400
CHART = pygame.Rect(0, GAME_H, WIN_W, WIN_H - GAME_H)

BG = (24, 26, 32)
PANEL_BG = (32, 35, 44)
TEXT = (225, 228, 235)
DIM = (140, 146, 160)
ACCENT = (255, 190, 70)      # greedy evaluation line
GREEN = (90, 210, 140)       # target line
BLUE = (110, 160, 255)       # training episodes


def fmt(x, digits=2):
    """Format a number, or '-' if we don't have one yet."""
    return "-" if x is None or np.isnan(x) else f"{x:.{digits}f}"


# ----------------------------------------------------------------- drawing
def draw_game(screen, env, m):
    """Ask Gymnasium for the current picture of the game and blit it."""
    frame = env.render()                                   # array shaped (height, width, 3)
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))   # pygame wants (width, height)
    screen.blit(surface, (0, 0))
    if m["status"]:
        banner = pygame.Rect(0, GAME_H // 2 - 22, GAME_W, 44)
        pygame.draw.rect(screen, (0, 0, 0), banner)
        label = m["fonts"]["big"].render(m["status"], True, ACCENT)
        screen.blit(label, label.get_rect(center=banner.center))


def draw_panel(screen, m, ui, cfg):
    """The metrics column to the right of the game."""
    x0 = GAME_W
    pygame.draw.rect(screen, PANEL_BG, (x0, 0, WIN_W - GAME_W, GAME_H))
    font, big = m["fonts"]["small"], m["fonts"]["big"]

    # Which stage of training are we in? (This is the story to tell in the demo.)
    if m["step"] < cfg.learning_starts:
        phase = "Collecting data"            # buffer filling, no gradient updates yet
    elif m["step"] < cfg.eps_decay_steps:
        phase = "Learning + exploring"        # epsilon still shrinking
    else:
        phase = "Learning, low epsilon"       # epsilon has reached its minimum
    screen.blit(big.render(phase, True, ACCENT), (x0 + 14, 12))

    reached = f"step {m['reached_step']:,}" if m["reached_step"] else "not yet"
    rows = [
        ("Step", f"{m['step']:,} / {m['total_steps']:,}"),
        ("Episode", f"{m['episode']}"),
        ("Epsilon (random %)", f"{m['epsilon']:.2f}"),
        ("Buffer size", f"{m['buffer_len']:,}"),
        None,
        ("This episode so far", f"{m['ep_return']:.0f}"),
        ("Last episode", fmt(m["last_ep_return"], 0)),
        ("Avg last 10 episodes", fmt(np.mean(m["recent"]) if m["recent"] else None, 1)),
        None,
        ("Loss", fmt(m["stats"]["loss"], 3)),
        ("Mean Q (taken action)", fmt(m["stats"]["q_mean"], 1)),
        None,
        ("Last greedy eval", fmt(m["last_eval"], 1)),
        ("Best greedy eval", fmt(m["best_eval"], 1)),
        (f"Reached {cfg.target_return:g}?", reached),
        None,
        ("Speed (steps/s)", f"{m['sps']:.0f}"),
    ]
    y = 46
    for row in rows:
        if row is None:
            y += 8
            continue
        screen.blit(font.render(row[0], True, DIM), (x0 + 14, y))
        value = font.render(row[1], True, TEXT)
        screen.blit(value, (WIN_W - 14 - value.get_width(), y))
        y += 20

    state = "PAUSED  " if ui["paused"] else ""
    hint = f"{state}speed {SPEEDS[ui['speed']][0]}"
    screen.blit(font.render(hint, True, ACCENT), (x0 + 14, GAME_H - 44))
    screen.blit(font.render("keys: 1/2/3 speed, SPACE pause, Q quit", True, DIM),
                (x0 + 14, GAME_H - 22))


def draw_chart(screen, m, cfg):
    """Learning curve: blue dots = training episodes, orange line = greedy evals."""
    pygame.draw.rect(screen, BG, CHART)
    left, right = CHART.x + 60, CHART.right - 20
    top, bottom = CHART.y + 28, CHART.bottom - 34
    font = m["fonts"]["small"]

    def to_xy(step, ret):
        x = left + (right - left) * step / m["total_steps"]
        y = bottom - (bottom - top) * max(ret, 0) / MAX_RETURN
        return x, y

    # axes and labels
    pygame.draw.line(screen, DIM, (left, top), (left, bottom))
    pygame.draw.line(screen, DIM, (left, bottom), (right, bottom))
    for ret in (0, 250, 500):
        label = font.render(str(ret), True, DIM)
        screen.blit(label, (left - 8 - label.get_width(), to_xy(0, ret)[1] - 8))
    for frac in (0, 0.5, 1):
        step = int(m["total_steps"] * frac)
        label = font.render(f"{step:,}", True, DIM)
        screen.blit(label, (to_xy(step, 0)[0] - label.get_width() / 2, bottom + 6))
    axis_label = font.render("online env steps", True, DIM)      # placed between the 0 and middle ticks
    screen.blit(axis_label, (left + (right - left) / 4 - axis_label.get_width() / 2, bottom + 6))

    # dashed target line
    ty = to_xy(0, cfg.target_return)[1]
    for x in range(int(left), int(right), 14):
        pygame.draw.line(screen, GREEN, (x, ty), (x + 7, ty))

    # training episodes (noisy, because they include random exploration)
    for step, ret in m["ep_points"]:
        pygame.draw.circle(screen, BLUE, to_xy(step, ret), 2)

    # greedy evaluation (the real measure of skill)
    points = [to_xy(s, r) for s, r in m["eval_points"]]
    if len(points) >= 2:
        pygame.draw.lines(screen, ACCENT, False, points, 2)
    for p in points:
        pygame.draw.circle(screen, ACCENT, p, 4)

    # legend
    legend_y = CHART.y + 8
    pygame.draw.circle(screen, BLUE, (left + 6, legend_y + 8), 3)
    screen.blit(font.render("training episode", True, DIM), (left + 16, legend_y))
    pygame.draw.circle(screen, ACCENT, (left + 170, legend_y + 8), 4)
    screen.blit(font.render("greedy evaluation", True, DIM), (left + 182, legend_y))
    screen.blit(font.render(f"- - target {cfg.target_return:g}", True, GREEN), (left + 350, legend_y))


def draw_all(screen, env, m, ui, cfg):
    draw_game(screen, env, m)
    draw_panel(screen, m, ui, cfg)
    draw_chart(screen, m, cfg)
    pygame.display.flip()


# ------------------------------------------------------------ keyboard/window
def handle_events(ui):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            ui["quit"] = True
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                ui["quit"] = True
            elif event.key == pygame.K_SPACE:
                ui["paused"] = not ui["paused"]
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                ui["speed"] = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3}[event.key]


# -------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="override total_steps")
    parser.add_argument("--speed", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--exit-when-done", action="store_true",
                        help="close the window when training ends (default: stay open)")
    args = parser.parse_args()

    cfg = get_config(ENV_ID)
    total_steps = args.steps or cfg.total_steps
    set_seed(args.seed)

    # render_mode="rgb_array" makes env.render() hand us the picture as an array,
    # so we can draw it inside our own window instead of Gymnasium opening one.
    env = gym.make(cfg.env_id, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, n_actions, cfg)
    buffer = ReplayBuffer(obs_dim, cfg.buffer_size)

    pygame.init()
    pygame.display.set_caption(f"DQN learning {ENV_ID} (seed {args.seed})")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    fonts = {"small": pygame.font.SysFont("consolas,couriernew,monospace", 16),
             "big": pygame.font.SysFont("consolas,couriernew,monospace", 20, bold=True)}

    ui = {"paused": False, "quit": False, "speed": args.speed}
    # Everything the drawing code needs to show lives in this one dict.
    m = {"fonts": fonts, "total_steps": total_steps, "step": 0, "episode": 0, "epsilon": 1.0,
         "buffer_len": 0, "ep_return": 0.0, "last_ep_return": None, "recent": deque(maxlen=10),
         "stats": {"loss": float("nan"), "td_loss": float("nan"), "q_mean": float("nan")},
         "last_eval": None, "best_eval": None, "reached_step": None,
         "status": "", "sps": 0.0, "ep_points": [], "eval_points": []}

    eval_seed = 10_000 + 100 * args.seed       # same rule as train_online.py
    obs, _ = env.reset(seed=args.seed)
    ep_len = 0
    last_draw_time, last_draw_step = time.time(), 0

    for step in range(1, total_steps + 1):
        # ---- exactly the training step from train_online.py ----
        epsilon = linear_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps, step)
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        m["ep_return"] += reward
        ep_len += 1

        if terminated or truncated:
            m["episode"] += 1
            m["last_ep_return"] = m["ep_return"]
            m["recent"].append(m["ep_return"])
            m["ep_points"].append((step, m["ep_return"]))
            obs, _ = env.reset()
            m["ep_return"], ep_len = 0.0, 0

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                m["stats"] = agent.update(buffer.sample(cfg.batch_size))

        if step % cfg.target_update == 0:
            agent.sync_target()

        m["step"], m["epsilon"], m["buffer_len"] = step, epsilon, len(buffer)

        if step % cfg.eval_every == 0:
            m["status"] = "Evaluating greedy policy..."     # shown while evaluate() runs
            draw_all(screen, env, m, ui, cfg)
            mean_ret, std_ret = evaluate(agent, cfg.env_id, cfg.eval_episodes, eval_seed)
            m["status"] = ""
            m["last_eval"] = mean_ret
            m["best_eval"] = mean_ret if m["best_eval"] is None else max(m["best_eval"], mean_ret)
            m["eval_points"].append((step, mean_ret))
            if m["reached_step"] is None and mean_ret >= cfg.target_return:
                m["reached_step"] = step
            print(f"step {step:>7} | episodes {m['episode']:>4} | eval {mean_ret:8.1f} ± {std_ret:6.1f}"
                  f" | eps {epsilon:.2f}")

        # ---- drawing, only every N steps so training is not slowed too much ----
        name, draw_every, max_fps = SPEEDS[ui["speed"]]
        if step % draw_every == 0 or step % cfg.eval_every == 0:
            handle_events(ui)
            while ui["paused"] and not ui["quit"]:           # paused: keep redrawing, no training
                draw_all(screen, env, m, ui, cfg)
                clock.tick(30)
                handle_events(ui)
                last_draw_time = time.time()
            if ui["quit"]:
                break
            now = time.time()
            if now > last_draw_time:
                m["sps"] = (step - last_draw_step) / (now - last_draw_time)
            last_draw_time, last_draw_step = now, step
            draw_all(screen, env, m, ui, cfg)
            if max_fps:
                clock.tick(max_fps)

    # Training finished: leave the final picture on screen until the user closes it.
    if not ui["quit"] and not args.exit_when_done:
        m["status"] = "Training finished (press Q to close)"
        while not ui["quit"]:
            draw_all(screen, env, m, ui, cfg)
            handle_events(ui)
            clock.tick(15)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
