# CLAUDE.md: handover notes for this project

Read this first. It gives the context, decisions and next steps for this
coursework project, so you can continue without re-deriving them.

## Who and what

- Student: Umer, BSc Computer Science with AI, University of Nottingham Malaysia.
  New to RL. Explain changes in plain language and keep the code readable.
- Module: Autonomous Robotics Systems (ARS) 26/27, reinforcement learning coursework.
- **Demo 1 (Task 2) is on 11 Nov 2026.** It is marked on one criterion:
  *"Evidence of understanding of the base code"*. The student must be able to
  explain every line, so:
  - build in small steps the student runs and checks;
  - comment the *why*, not just the *what*;
  - prefer plain, explicit code over clever abstractions or extra libraries;
  - after each change, summarise what changed and why, in a few sentences.
- Final submission (Task 3) is on 16 Dec 2026. Software is marked (20%) on
  completeness, clean design, comments and enough complexity.
- **Academic integrity:** the guide treats uncited AI use as plagiarism. Never
  claim the work is AI-free. The student acknowledges AI assistance in their write-ups.

## The research (Task 1, already submitted)

- Core paper: **ActiveRL**, Dukkipati et al., AAAI 2025, *Active Reinforcement
  Learning Strategies for Offline Policy Improvement*. It spends a limited online
  budget on uncertain states:
  1. starting episodes from uncertain states;
  2. taking the most uncertain candidate action with probability ε;
  3. ending a trajectory (termination function β) once a state looks familiar.

  Uncertainty is measured as disagreement within an ensemble of networks.
- Supporting paper: Roy, Kushwaha & Dukkipati 2026, arXiv 2602.01260 (GP theory
  of ActiveRL; continuous tasks only).
- Gaps chosen:
  - **G1:** only continuous actions have been tested;
  - **G2:** the termination trade-off (few long vs many short trajectories) has never been ablated;
  - **G3:** novelty is assumed useful, but untested where unfamiliar states mean failure.
- **Submitted novelty statement (do not change its meaning):**
  > Dukkipati et al. (2025) demonstrate that offline-to-online RL is improved by
  > allocating a small online budget to uncertain states; nevertheless, they only
  > test continuous-action tasks and never assess the trajectory-termination
  > trade-off, which they see as a major obstacle. In order to determine if
  > novelty-seeking still outperforms naive fine-tuning when unusual states result
  > in failure (CartPole) rather than progress (Acrobot), this work extends
  > uncertainty-guided data collection to discrete control (CartPole and Acrobot),
  > assessing how termination length impacts sample efficiency.
- Research questions:
  - **RQ1 (G1):** Does uncertainty-guided collection reduce the online interactions
    needed to reach the target return, compared with naive O2O and pure online?
    Metrics: interactions to target, % saved vs naive, final return.
  - **RQ2 (G2):** Under a fixed online budget, how does the termination setting
    affect sample efficiency? Metrics: final return, area under the learning curve,
    trajectories collected.
  - **RQ3 (G3):** Does the active-vs-naive gap differ between CartPole
    (failure-dominated) and Acrobot (progress-dominated)? Vary ε ∈ {0, 0.2, 0.5}.
    Metrics: gap per env, failure rate of collected data, return vs ε.
- Fixed across conditions: offline dataset, network size, online budget and
  evaluation protocol. Use ≥5 seeds, report mean ± 95% CI.

## Coursework constraints (from the guide)

- Environment 1 must be a control task, and environment 2 must relate to robotics.
  The choice is fixed: **CartPole-v1** and **Acrobot-v1** (a two-link robot arm).
  Both are discrete. Task 3 must use the same two.
- The setup must test all three configurations: **(1) pure online, (2) pure
  offline, (3) offline-online**.
- Task 2 aim: 3 proofs of concept per environment, 6 in total.
- Example metrics in the guide: convergence speed (episodes to reach a target)
  and average reward. We log **both env steps and episodes**.
- Task 3 needs at least one online or hybrid comparison condition.

## Current state (6 Nov 2026)

Done: all three configurations (**1 pure online DQN, 2 offline CQL, 3 naive offline-to-online**),
the **dataset generator**, and `summarise.py`, on both environments with 3 seeds each. Still to do:
Task 3 (active collection) and the Demo 1 walkthrough practice (see "Next steps").
Everything about results is regenerated in `results/RESULTS.md` by `python summarise.py`
(never edit RESULTS.md by hand); the tables below are a copy from 6 Nov 2026.

```
rl/config.py         per-env hyperparameters (DQNConfig dataclass, CONFIGS dict)
rl/networks.py       QNetwork: MLP, 2 hidden layers
rl/replay_buffer.py  circular buffer; save()/load() to .npz = offline datasets
rl/dqn.py            DQNAgent: eps-greedy, Double DQN targets, Huber loss + CQL penalty
                     (weight cfg.cql_alpha, 0.0 = off), grad clipping, hard target sync.
                     update() returns loss, td_loss, cql, q_mean.
rl/evaluate.py       greedy eval on a separate env with fixed seeds
rl/utils.py          set_seed (also torch.set_num_threads(1)), linear_schedule, CSVLogger
rl/paths.py          run_dir(mode, env, seed, dataset) and find_runs(): the ONE place that defines the
                     results/ folder layout (see "Results layout" below)
train_online.py      the training loop; writes eval.csv, episodes.csv, summary.json, model.pt into
                     results/online/<env>/s<seed>/; --save-buffer writes data/..._buffer.npz
plot.py              quick single-env learning curves (online + O2O): mean, min-max band, target line
summarise.py         all three configurations: tables, figures, results/RESULTS.md (see below)
tools/migrate_results.py  one-off: moved the old flat results/ files into the new layout (done)
watch_online.py      live pygame view of CartPole online training (game, metrics, learning
                     curve; keys 1/2/3 speed, SPACE pause, Q quit). Same loop and seeding as
                     train_online.py; writes no files. A learning/demo aid, not an experiment.
make_dataset.py      writes data/<env>_<random|medium>_<N>.npz (+ .json with the average
                     episode return and how it was made). See "Datasets" below.
train_offline.py     Configuration 2: loads a dataset into a ReplayBuffer, runs gradient
                     updates with the CQL loss and NO game steps (the game is only used by
                     evaluate()). Writes results/offline/<env>/<policy>/s<seed>/{eval.csv,summary.json,
                     model.pt}. The eval CSV's x-axis column is `gradient_steps`.
train_o2o.py          Configuration 3 (naive offline-to-online): CQL weights + dataset in a big buffer,
                     then epsilon-greedy online play. Loads results/offline/<env>/<policy>/s<seed>/model.pt (the
                     MATCHING seed). Writes results/o2o/<env>/<policy>/s<seed>/.
summarise_offline.py  table (mean ± SD over seeds of final/best return, last mean Q) per env and
                     dataset + results/figures/offline_curves.png overlaying the seeds' eval curves.
```

DQNConfig also has dataset fields: `dataset_size`, `medium_low`, `medium_high`,
`medium_check_every`, `medium_epsilon` (used only by `make_dataset.py`), and CQL/offline
fields: `cql_alpha` (0.0 default, so every online run is plain DQN), `offline_alpha` (1.0,
what train_offline.py uses), `offline_steps` (20,000), `offline_eval_every` (1,000),
`offline_target_update` (250; CartPole overrides to 128 = one online burst).

Other changes made with CQL: `train_online.py`'s eval CSV gained a `cql` column (update()
now returns that key). O2O fields: `o2o_alpha` (0.0, CQL weight while playing online) and
`o2o_eps_start` (0.1, starting epsilon, decaying to `eps_end` over `eps_decay_steps`).

## Results layout (since 6 Nov 2026)

```
results/online/<env>/s<seed>/              eval.csv, episodes.csv, summary.json, model.pt
results/offline/<env>/<dataset>/s<seed>/   eval.csv, summary.json, model.pt
results/o2o/<env>/<dataset>/s<seed>/       eval.csv, episodes.csv, summary.json, model.pt
results/figures/                           every .png (curves_<env>.png, offline_curves.png)
results/archive/                           old/superseded runs, kept but never used in tables
results/RESULTS.md                         generated by summarise.py (never edit by hand)
logs/                                      console output of batch runs (+ _jobs.txt, _run_jobs.sh)
```

All paths come from `rl/paths.py`. A run counts as finished once its summary.json exists.
**Archived:** `results/archive/o2o_seed0_first_run/` = the first O2O seed-0 runs (5 Nov session), replaced by a
rerun in the same session as seeds 1-2; the rerun gave identical numbers (steps to target and final return
matched on all four env/dataset pairs). `results/archive/figures_quick_plot/` = the older plot.py figures,
replaced by `curves_<env>.png`. The alpha-0 throwaway runs were never in results/.
`results/archive/online_4nov/<env>/s<seed>/` = the four online runs made on 4 Nov (CartPole s0-s2, Acrobot s0),
archived and RERUN on 6 Nov with unchanged code and settings so every online run is from one session. **Outcome:
the reruns are identical** on all four: steps to target, episodes to target, final return, last-5 mean and last
mean Q all matched, and eval_return, eval_std, epsilon, loss, td_loss, q_mean differ by exactly 0 in every row
(max |diff| of eval_return = 0.0). The eval.csv files are not byte-identical only because the 4 Nov files predate
the `cql` column (the shared columns are exactly equal). All online and O2O runs in the tables are from 6 Nov.

All three configurations, mean ± sample SD over 3 seeds (0, 1, 2), from `python summarise.py`
(6 Nov 2026; budgets, eval_every, epsilon and alpha unchanged). O2O seed k starts from offline seed k.
Last-mean-Q sanity limits: CartPole <= 100, Acrobot <= 0. Trust the mean of the last 5 evals, then final;
best eval and steps-to-target are optimistic (the first crossing can be a lucky spike).

**CartPole-v1 (target 475)**

| Configuration | Online steps used | Steps to target | Episodes to target | Final return (20 eps) | Mean of last 5 evals | Best eval | Last mean Q | Seeds |
|---|---|---|---|---|---|---|---|---|
| online | 50000 ± 0 | 28000 ± 10583 | 423 ± 190 | 111.7 ± 136.3 | 276.3 ± 89.7 | 500.0 ± 0.0 | 82.0 ± 7.1 | 3 |
| offline CQL, random data | 0 | n/a (gradient updates) | n/a | 459.4 ± 70.3 | 418.6 ± 76.5 | 500.0 ± 0.0 | 30.1 ± 1.3 | 3 |
| offline CQL, medium data | 0 | n/a (gradient updates) | n/a | 246.4 ± 16.5 | 248.9 ± 0.9 | 274.2 ± 0.5 | 61.3 ± 1.2 | 3 |
| O2O, random data | 50000 ± 0 | 667 ± 1155 | 1 ± 2 | 69.5 ± 13.5 | 269.6 ± 155.6 | 500.0 ± 0.0 | 110.1 ± 17.9 | 3 |
| O2O, medium data | 50000 ± 0 | 3333 ± 1155 | 44 ± 30 | 500.0 ± 0.0 | 500.0 ± 0.0 | 500.0 ± 0.0 | 99.7 ± 5.5 | 3 |

**Acrobot-v1 (target -100)**

| Configuration | Online steps used | Steps to target | Episodes to target | Final return (20 eps) | Mean of last 5 evals | Best eval | Last mean Q | Seeds |
|---|---|---|---|---|---|---|---|---|
| online | 100000 ± 0 | 27333 ± 10263 | 173 ± 91 | -79.5 ± 3.0 | -79.6 ± 5.9 | -68.5 ± 0.7 | -30.8 ± 1.4 | 3 |
| offline CQL, random data | 0 | n/a (gradient updates) | n/a | -488.0 ± 20.8 | -485.1 ± 21.8 | -320.8 ± 112.7 | -52.7 ± 0.4 | 3 |
| offline CQL, medium data | 0 | n/a (gradient updates) | n/a | -250.0 ± 71.0 | -294.9 ± 55.3 | -202.5 ± 44.9 | -50.8 ± 0.7 | 3 |
| O2O, random data | 100000 ± 0 | 6000 ± 0 | 40 ± 7 | -77.7 ± 2.0 | -72.8 ± 0.6 | -69.2 ± 2.0 | -33.3 ± 2.1 | 3 |
| O2O, medium data | 100000 ± 0 | 17333 ± 3055 | 95 ± 15 | -96.0 ± 41.4 | -79.6 ± 10.6 | -68.9 ± 2.5 | -32.3 ± 0.6 | 3 |

- **CartPole online collapses late** (final 111.7 vs last-5 mean 276.3; each seed reached 500 once). Cause
  not found; not tuned.
- **CartPole O2O random's "667 ± 1,155 steps to target" is NOT a speed-up.** On 2 of 3 seeds (s1, s2) the offline
  CQL agent already scored >= 475 at online step 0 (steps to target = 0); only seed 0 (395.4 at step 0) needed
  2,000 online steps. The mean mixes "already solved" with one real run. CartPole random O2O then ends at
  69.5 ± 13.5 with last mean Q 110.1 ± 17.9, above the 100 ceiling (overestimation seen, not proven as the
  cause). CartPole medium O2O (start 238-260, so a real learning gain) needed 3,333 ± 1,155 online steps vs
  28,000 ± 10,583 for pure online and held 500 on every seed.
- **Acrobot O2O uses epsilon 0.1 throughout** (`o2o_eps_start` 0.1 = Acrobot's `eps_end` 0.1), while pure online
  decays from 1.0. Part of O2O's early advantage may come from less random play, not only from the data. An
  ablation (O2O with epsilon decaying from 1.0, and/or dataset without CQL weights) would separate the two; not run.
- **Fine-tuning dip** (online-step-0 eval vs lowest in first 6,000 steps; full table in RESULTS.md): Acrobot
  medium fell from -219..-326 to -500 at step 4,000 on all 3 seeds, then recovered. CartPole medium dipped on
  2 of 3 seeds (-47.5, -29.5); CartPole random on 1 of 3 (-11.4). "No dip" = lowest early eval >= step 0.
- Acrobot random O2O starts at -500 on every seed (offline learned nothing) yet reaches the target after
  6,000 steps on all seeds.
- Online Acrobot is 3 seeds (s0 rerun on 6 Nov, identical to 4 Nov): 27,333 ± 10,263 steps to target, final -79.5 ± 3.0 (stable).

Key decisions and why:
- **Terminated vs truncated:** only `terminated` stops bootstrapping. Time-limit
  truncation still bootstraps from the next state.
- **Double DQN** is on by default (van Hasselt et al., 2016) to reduce
  overestimation.
- **Hyperparameters come from RL Baselines3 Zoo** (Raffin, 2020). For CartPole,
  updating every step plateaued at a return of 100–200. Burst training (128 updates
  every 256 steps, target sync every 10 steps) reaches 500. Acrobot uses the Zoo
  values, untuned here (3 seeds run, stable).
- **Convergence speed** = the first evaluation where the greedy return is ≥ target
  (475 CartPole, −100 Acrobot). Final return is logged separately because DQN can
  collapse after reaching the target (all three CartPole seeds did).
- **Evaluation:** 10 greedy episodes every `eval_every` steps; 20 at the end.
  Evaluation seeds are offset from training seeds.
- **Single-thread torch:** run at most ~(cores) seeds in parallel. All runs from 4 Nov onward are on this
  laptop (8 cores); the 1 Nov runs came from a different machine (the cloud dev box, 2 cores).
- **CartPole late-training instability (observed, cause not proven):**
  - The CartPole buffer never wraps (50k steps, capacity 100k), so forgetting through
    overwriting is not the explanation.
  - With +1 reward and gamma 0.99 no state can be worth more than 1/(1-0.99) = 100.
    In the EARLIER (overwritten) seed-2 run the mean Q was 127-138, above that ceiling,
    i.e. overestimated. In the 4 Nov runs the last mean Q was 76-90 (below it), yet the
    seeds still collapsed, so overestimation alone does not explain it.
  - "First eval >= target" can reward a lucky spike. Keep this in mind for Task 3 metrics.
- **Reproducibility:** the very first seed-0 run (1 Nov, cloud dev box, 2 cores) reached the target at
  24,000 steps with final 500; this laptop gives 32,000 steps and final 267.9 for the same command. That gap is
  between DIFFERENT MACHINES (and possibly library builds), not between sessions. On this laptop results DO
  reproduce exactly: the four online runs rerun on 6 Nov matched the 4 Nov runs to the last digit, and O2O seed 0
  rerun matched its first run on all four env/dataset pairs; `watch_online.py` also matched `train_online.py`.
  Do not expect to match numbers from another machine. For the final experiments report mean ± CI over >= 5 seeds
  rather than any single run.

## Datasets (made by `make_dataset.py`, seed 0, N = 20,000 transitions each)

Each `.npz` holds the five buffer fields (obs, actions, rewards, next_obs, terminated).
The `.json` beside it holds the average episode return (needed for Task 3).

| Dataset | Finished episodes | Average return | How episodes ended |
|---|---|---|---|
| CartPole random | 897 | 22.3 ± 11.6 | all 897: pole fell |
| CartPole medium | 101 | 196.1 ± 84.9 | all 101: pole fell, none reached 500 |
| Acrobot random | 40 | -496.3 ± 23.3 | 1 reached the goal, 39 hit the time limit |
| Acrobot medium | 102 | -194.3 ± 71.1 | 101 reached the goal, 1 hit the limit |

- **"Medium" agent:** an ordinary online DQN, stopped the first time its greedy test
  score reaches `medium_low` (30%) of the way from random play to the target return
  (random play measured over 5,000 steps). Data is then collected with that frozen agent,
  epsilon 0.05. "Fraction of the way" is used because Acrobot scores are negative.
- **Acrobot has no gradual middle:** the score jumped from -500 to -186 within 100 steps on
  one seed, so a 30-60% band can be skipped. The script therefore stops at the first
  check above 30% and reports where the agent really was. `medium_high` is only a warning.
- **Medium-ness is loose:** CartPole medium agent tested at 48% (data: 38%); Acrobot medium
  agent tested at 51% but its data averages 76%. Both numbers are in the JSON
  (`agent_progress`, `dataset_progress`). Cause of the Acrobot gap not tested.
- **Acrobot random data has almost no successes** (1 of 40 episodes), so offline learning
  on it will be hard. Remember this when reading CQL results.
- **`terminated` means different things:** CartPole, the pole fell (failure). Acrobot, the
  goal was reached (success). Matters for RQ3's "failure rate of collected data".
- Datasets are one sample from seed 0; other seeds or sizes are untested.

## Offline results, 3 seeds (CQL alpha = 1.0, 20,000 gradient updates, seed-0 datasets)

Training seeds 0, 1, 2 on the SAME dataset per env/policy (so this measures training noise,
not dataset noise). Made by `python summarise_offline.py`, which also saves
`results/figures/offline_curves.png` (one panel per env and dataset, one line per seed).
Mean ± sample SD (n = 3). "Final" = 20 greedy episodes after the last update; "best" = highest
of the 20 periodic 10-episode tests (a lucky-spike measure, so optimistic); "last mean Q" =
average Q of the dataset's actions at the last logged update.

| Env | Dataset (avg return) | Final return | Best return | Last mean Q |
|---|---|---|---|---|
| CartPole | random (22.3) | 459.4 ± 70.3 | 500.0 ± 0.0 | 30.1 ± 1.3 |
| CartPole | medium (196.1) | 246.4 ± 16.5 | 274.2 ± 0.5 | 61.3 ± 1.2 |
| Acrobot | random (-496.3) | -488.0 ± 20.8 | -320.8 ± 112.7 | -52.7 ± 0.4 |
| Acrobot | medium (-194.3) | -250.0 ± 71.0 | -202.5 ± 44.9 | -50.8 ± 0.7 |

- Seed 0 reran with identical numbers to the earlier single-seed run (same session).
- CartPole random reaches 500 on all 3 seeds but the curve oscillates (seed 0 ended at 378, others 500).
- CartPole medium sits at ~240-275 on every seed and never reaches 475: CQL stays close to the
  data's behaviour (average 196) and does not climb past it. Not investigated.
- Acrobot random is essentially a failure (final near -500; only brief spikes), as expected from
  the data (1 success in 40 episodes). Acrobot medium improves but stays far from -100.
- Wall time 190-207 s per CartPole run, 246-274 s per Acrobot run (6 in parallel).
- Offline learning uses zero env steps (online_steps_used = 0).

### Earlier single-seed run (seed 0) with the alpha = 0 comparison

CQL loss: `cql = (torch.logsumexp(q_all, dim=1) - q_taken).mean(); loss = td_loss + alpha * cql`
(Kumar et al., 2020, Eq. 4). Zero game steps are used for learning, so "online steps used" is 0.
The alpha = 0 columns are plain offline DQN, run once as a comparison (thrown away, not in
`results/`; regenerate with `--alpha 0 --out <folder>`).

| Run | Data avg return | Updates to target | Best eval | Final (20 eps) | Last mean Q | alpha 0: final (best), last mean Q |
|---|---|---|---|---|---|---|
| CartPole random | 22.3 | 15,000 | 500.0 | 378.2 | 30.5 | 101.2 (374.5), 105.7 |
| CartPole medium | 196.1 | not reached | 273.9 | 235.7 | 60.8 | 9.3 (9.5), **705,674** |
| Acrobot random | -496.3 | not reached | -438.1 | -500.0 | -52.4 | -104.1 (-101.7), -53.5 |
| Acrobot medium | -194.3 | not reached | -174.1 | -330.5 | -50.6 | -99.0 (-158.5), -33.8 |

- **CQL clearly helped on CartPole.** Without it, offline DQN on the medium data diverged
  (mean Q 705,674, score 9.5) and on random data its Q (105.7) went above the maximum possible
  value of 100 and the score stayed near 100. With alpha 1.0 the Q values stayed at 30-61 and
  CartPole random data (average return 22!) reached 500 at times.
- **On Acrobot alpha 1.0 did NOT help in these single runs:** it ended worse than alpha 0
  (-330 vs -99 on medium, -500 vs -104 on random). But alpha 0 was unstable (e.g. four
  evals at -500 then -174), so its good final scores may be luck. alpha is untuned and
  there is one seed. Cause not investigated; a possible reading (not tested) is that alpha
  1.0 is too conservative when rewards are small (-1 per step) and the data has few successes.
- **Offline scores swing between evals** (CartPole random: 199, 444, 500, 500, 395). Report the
  final value, the best value and the curve, never one number.
- "Updates to target" is counted in gradient updates and is NOT comparable with online
  "steps to target" (game steps). Compare final return, and the online budget used (0 vs N).
- CartPole medium alpha 1.0 peaked at its first eval (273.9) then drifted to ~238 while mean
  Q kept rising (26 to 61). Not investigated.

## Offline-to-online (naive O2O, `train_o2o.py`)

Starts from the matching-seed CQL weights. Buffer = dataset + online data (capacity dataset + budget, so the
dataset is never overwritten; asserted in the script). Epsilon 0.1 decaying to eps_end, online CQL alpha 0, same
schedule as pure online but no `learning_starts` wait. Steps to target count ONLINE steps only; eval row 0 is the
offline agent before any play. Results are in the tables above and in `results/RESULTS.md`.

## Next steps for Demo 1 (in order)

1. ~~Dataset generator~~ **DONE** (`make_dataset.py`).
2. ~~CQL offline~~ **DONE** (`train_offline.py`, 3 seeds).
3. ~~Naive offline-to-online~~ **DONE** (`train_o2o.py`, 3 seeds, both envs).
4. ~~All three configurations on Acrobot~~ **DONE** (6 proofs of concept, 3 seeds each).
5. ~~summarise.py + RESULTS.md + overlay figures~~ **DONE** (`python summarise.py`).
6. Practise the walkthrough: loop → targets → loss → buffer save/load → CQL term → O2O (capacity trap).
7. Open issues: CartPole late collapse / Q above 100 (O2O random); coarse eval_every (2,000 steps) hides
   spikes, to be revisited for Task 3; one dataset (seed 0) only; results reproduce exactly on this laptop but not against the 1 Nov cloud-box run.

## Later (Task 3, after Demo 1)

- Active O2O collection (ActiveRL-style), replacing the naive collector in step 3:
  - an ensemble (about 5 small networks) whose disagreement gives uncertainty;
  - uncertain start states through `env.unwrapped.state = ...` (classic control
    allows this; set it after `reset()`);
  - with probability ε, take the most uncertain action (for discrete actions,
    score every action rather than sampling);
  - a termination rule β with a threshold or max-length setting (this is the RQ2 knob).
- Experiments for RQ1–RQ3 with ≥5 seeds, 95% CIs, the same dataset and budget.

## Conventions

- Keep the shared `rl/` package; one script per configuration at the top level.
- Results go to `results/` via `rl/paths.py` (never hand-build result paths), datasets to `data/`.
  Don't commit `.pt` files. After new runs, rerun `python summarise.py`.
- Every new hyperparameter goes in `DQNConfig` with a comment.
- Test-run before claiming something works, and report the actual numbers.
