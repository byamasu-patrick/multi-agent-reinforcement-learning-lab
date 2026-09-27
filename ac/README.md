# Multi-Agent Actor-Critic: IA2C, MAA2C, IPPO, MAPPO

A standalone package of four multi-agent actor-critic algorithms, implemented in PyTorch and configured with [Hydra](https://hydra.cc/). The code started as the `ac` module of the [MARL book codebase](https://github.com/marl-book/codebase) (Albrecht, Christianos and Schäfer, 2024). It has been separated from that codebase so it runs on its own and can be studied, changed and extended here.

## Table of contents

- [The four algorithms](#the-four-algorithms)
- [How the update works](#how-the-update-works)
- [Setup](#setup)
- [Training](#training)
- [Outputs](#outputs)
- [Recording a trained agent](#recording-a-trained-agent)
- [Configuration reference](#configuration-reference)
- [Code map](#code-map)
- [Changes from the MARL book codebase](#changes-from-the-marl-book-codebase)

## The four algorithms

All four share one implementation. They differ in two independent choices:

1. **The critic's input.** A *decentralised* critic sees only agent $i$'s observation, $V_{\phi_i}(o_i)$. A *centralised* critic sees all agents' observations, $V_{\phi_i}(o_1, \dots, o_N)$.
2. **The policy update.** Either the vanilla advantage actor-critic (A2C) loss, or PPO's clipped loss applied for several epochs on each batch.

|  | Decentralised critic $V(o_i)$ | Centralised critic $V(o_1, \dots, o_N)$ |
|---|---|---|
| **A2C update** | IA2C | MAA2C |
| **PPO update** | IPPO | MAPPO |

The actors are always decentralised: agent $i$ acts from its own observation, $\pi_{\theta_i}(a_i \mid o_i)$. The centralised variants are therefore **centralised training with decentralised execution (CTDE)**. The extra information is used only by the critic during training, and each agent can still act alone.

In the code, the critic's input is `critic.centralised` in the config, and the update is `ac.model.A2CNetwork` or `ac.model.PPONetwork`.

## How the update works

Each training iteration in [`train.py`](train.py) does two things:

1. **Collect.** Run `parallel_envs` environments (10 by default) for one full episode each. Store observations, actions, rewards and done flags as tensors of shape `(time, env, agent)`.
2. **Update.** Call `model.update(batch)` once on that batch.

### Returns and advantages

For each agent $i$, the critic target is the $n$-step return (default $n = 5$). It bootstraps from a **target critic** $V_{\bar\phi_i}$, a periodically copied version of the critic:

$$G_{i,t}^{(n)} = \sum_{k=0}^{n-1} \gamma^k \, r_{i,t+k} + \gamma^n \, V_{\bar\phi_i}(s_{t+n})$$

Terms after the end of an episode are masked to zero. The advantage is

$$A_{i,t} = G_{i,t}^{(n)} - V_{\phi_i}(s_t)$$

where $s_t$ is $o_{i,t}$ for a decentralised critic and $(o_{1,t}, \dots, o_{N,t})$ for a centralised one.

### A2C loss (IA2C, MAA2C)

$$\mathcal{L} = \underbrace{\sum_i \Big( -\log \pi_{\theta_i}(a_{i,t} \mid o_{i,t}) \, A_{i,t} \Big) - \beta \sum_i \mathcal{H}\big(\pi_{\theta_i}(\cdot \mid o_{i,t})\big)}_{\text{actor loss}} + c_v \underbrace{\sum_i A_{i,t}^2}_{\text{value loss}}$$

The advantage is detached in the actor loss, so the actor's gradient does not flow into the critic. $\beta$ is `entropy_coef`, which rewards exploration, and $c_v$ is `value_loss_coef`. All losses are averaged over the valid (non-padded) time steps, and one gradient step is taken per batch.

### PPO loss (IPPO, MAPPO)

PPO first records the log-probabilities under the policy that collected the data, $\pi_{\theta_i^{\text{old}}}$. It then takes `num_epochs` gradient steps on the same batch. Each step uses the probability ratio

$$\rho_{i,t} = \frac{\pi_{\theta_i}(a_{i,t} \mid o_{i,t})}{\pi_{\theta_i^{\text{old}}}(a_{i,t} \mid o_{i,t})}$$

and replaces the A2C policy term with the clipped objective

$$-\min\Big( \rho_{i,t} A_{i,t}, \; \operatorname{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon) \, A_{i,t} \Big)$$

where $\epsilon$ is `ppo_clip`. Clipping stops a single batch from moving the policy too far. That makes reusing the batch for several epochs safe.

### Target critic

With `target_update_interval_or_tau > 1`, the target critic is overwritten with the critic's weights whenever the environment step count is a multiple of that value. With a value below 1, it is instead updated softly every iteration: $\bar\phi \leftarrow (1 - \tau)\bar\phi + \tau\phi$.

### Parameter sharing

`actor.parameter_sharing` and `critic.parameter_sharing` control whether agents share network weights:

- `False` gives each agent its own network.
- `True` makes all agents share one network.
- A list such as `[0, 0, 1]` shares within groups of agents (selective parameter sharing, SePS).

Sharing is often combined with `env.observe_id=True`, which appends a one-hot agent ID to each observation so a shared network can still tell agents apart.

## Setup

This package needs **Python 3.10–3.12** because `torch==2.4.1` and `pandas==2.2.2` have no wheels for newer versions. From this `ac/` folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

This installs the CPU build of PyTorch and [Level-Based Foraging](https://github.com/uoe-agents/lb-foraging) (LBF), the environment used in the examples below. Any Gymnasium environment works if it:

- returns one reward per agent as a list,
- has `Tuple` observation and action spaces,
- exposes `env.unwrapped.n_agents`.

## Training

Run from this `ac/` folder with the venv activated. `+algorithm=` chooses one of `ia2c`, `maa2c`, `ippo` or `mappo`, and `env.name` and `env.time_limit` are required:

```powershell
python run.py +algorithm=ia2c env.name="lbforaging:Foraging-8x8-2p-3f-v3" env.time_limit=25
```

The `lbforaging:` prefix tells Gymnasium to import the `lbforaging` package, which registers its environments. The ones installed are `Foraging-5x5-2p-1f-v3` and `Foraging-8x8-2p-3f-v3`, each with a fully cooperative `-coop-v3` version.

Any config value can be overridden on the command line:

```powershell
python run.py +algorithm=mappo env.name="lbforaging:Foraging-8x8-2p-3f-coop-v3" env.time_limit=25 `
    algorithm.total_steps=1_000_000 algorithm.lr=5e-4 algorithm.entropy_coef=0.01 seed=0
```

To compare algorithms or seeds, use Hydra's multirun (`-m`), which runs every combination one after another:

```powershell
python run.py -m +algorithm=ia2c,maa2c,ippo,mappo seed=0,1,2 env.name="lbforaging:Foraging-8x8-2p-3f-v3" env.time_limit=25
```

At the default 100,000 steps, one run takes a few minutes on a laptop CPU. A 20,000-step IA2C run on `Foraging-8x8-2p-3f-v3` took 18 seconds at about 4,800 environment steps per second.

## Outputs

Each run gets its own folder:

```text
outputs/<env>/<algorithm>/<run-id>/            # single run
multirun/<env>/<timestamp>/<algorithm>/seed<seed>_<n>/   # multirun
├── config.yaml     # the full resolved config; evaluate.py reads it back
├── results.csv     # one row per eval_interval
├── run.log         # the progress printed during training
├── checkpoints/    # model_s<step>.pt, if algorithm.save_interval is set
└── videos/         # step-<step>.mp4, if algorithm.video_interval is set
```

`:` in environment names becomes `_` in folder names, because Windows does not allow colons in paths.

`results.csv` has one row per logging interval. The main columns are:

- `environment_steps` and `updates`
- `mean_episode_returns` and `std_episode_returns`, summed over agents
- `agent<i>/mean_episode_returns`, per agent
- `actor_loss`, `value_loss` and `entropy`

To plot a learning curve:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/lbforaging_Foraging-8x8-2p-3f-v3/ia2c/<run-id>/results.csv")
plt.plot(df["environment_steps"], df["mean_episode_returns"])
plt.fill_between(
    df["environment_steps"],
    df["mean_episode_returns"] - df["std_episode_returns"],
    df["mean_episode_returns"] + df["std_episode_returns"],
    alpha=0.3,
)
plt.xlabel("Environment steps")
plt.ylabel("Mean episode return")
plt.show()
```

## Recording a trained agent

Train with checkpoints enabled, then point `evaluate.py` at the run folder:

```powershell
python run.py +algorithm=mappo env.name="lbforaging:Foraging-8x8-2p-3f-v3" env.time_limit=25 algorithm.save_interval=50_000
python evaluate.py path=outputs/lbforaging_Foraging-8x8-2p-3f-v3/mappo/<run-id> video_frames=500
```

This loads the latest checkpoint and writes `evals/<id>/eval.mp4`. To load a specific checkpoint, use `load_step=<step>`. LBF opens a small render window while it records.

## Configuration reference

Defaults live in [`configs/default.yaml`](configs/default.yaml), with algorithm settings in [`configs/algorithm/`](configs/algorithm/). The four algorithm files are identical apart from `name`, the model class and `critic.centralised`.

| Key | Default | Meaning |
|---|---|---|
| `seed` | `null` | Random seed. Set it for reproducible runs |
| `env.parallel_envs` | 10 | Environments run in parallel sub-processes per batch |
| `env.observe_id` | False | Append a one-hot agent ID to each observation |
| `env.standardise_rewards` | False | Normalise rewards with running statistics |
| `algorithm.total_steps` | 100,000 | Environment steps to train for, summed over parallel envs |
| `algorithm.eval_interval` | 10,000 | Steps between rows in `results.csv` |
| `algorithm.save_interval` | False | Steps between checkpoints |
| `algorithm.video_interval` | False | Steps between training videos |
| `algorithm.lr` | 3e-4 | Adam learning rate |
| `algorithm.gamma` | 0.99 | Discount factor $\gamma$ |
| `algorithm.n_steps` | 5 | $n$ in the $n$-step return |
| `algorithm.entropy_coef` | 0.001 | Entropy bonus weight $\beta$ |
| `algorithm.value_loss_coef` | 0.5 | Value loss weight $c_v$ |
| `algorithm.grad_clip` | False | Maximum gradient norm, or False for none |
| `algorithm.standardise_returns` | False | Normalise returns with running statistics |
| `algorithm.target_update_interval_or_tau` | 200 | Hard-update interval if > 1, soft-update rate $\tau$ if < 1 |
| `algorithm.num_epochs` | 4 | PPO epochs per batch (PPO only) |
| `algorithm.ppo_clip` | 0.2 | PPO clip range $\epsilon$ (PPO only) |
| `algorithm.model.actor.layers` | [128, 128] | Hidden layer sizes of each actor |
| `algorithm.model.critic.layers` | [128, 128] | Hidden layer sizes of each critic |
| `algorithm.model.*.use_rnn` | False | Add a GRU layer for partially observable tasks |
| `algorithm.model.device` | "cpu" | PyTorch device |

To log to Weights & Biases instead of CSV, add `logger=wandb`. This needs `pip install wandb` and a `wandb login`.

## Code map

```text
ac/
├── run.py                # entry point: builds envs and logger, then calls the algorithm's train.main
├── evaluate.py           # entry point: loads a run's config and checkpoint, then calls eval.main
├── train.py              # training loop: _collect_trajectories() -> model.update() -> logging
├── eval.py               # rebuilds the model from a checkpoint and records a video
├── model.py              # A2CNetwork (A2C loss) and PPONetwork (PPO loss)
├── configs/
│   ├── default.yaml      # env, logging and output-folder settings
│   ├── eval.yaml
│   ├── algorithm/        # ia2c / maa2c / ippo / mappo
│   ├── logger/           # basic (console), filesystemlogger (CSV, default), wandb
│   └── hydra/job_logging/
└── utils/
    ├── envs.py           # make_env(): gym.make + wrappers, vectorised if parallel_envs is set
    ├── wrappers.py       # RecordEpisodeStatistics, ObserveID, StandardiseReward, CooperativeReward, ...
    ├── models.py         # per-agent MLP/GRU networks, with or without parameter sharing
    ├── utils.py          # MultiCategorical (one distribution per agent) and compute_nstep_returns
    ├── standardise_stream.py  # running mean and variance for return standardisation
    ├── loggers.py        # console, CSV and W&B loggers
    └── video.py          # VideoRecorder
```

A good reading order is `train.py` → `model.py` (`A2CNetwork.update`, then `PPONetwork.update`) → `utils/utils.py` (`compute_nstep_returns`) → `utils/models.py`.

## Changes from the MARL book codebase

The algorithm code in `train.py`, `model.py` and `utils/` (models, returns, standardisation) is unchanged apart from import paths. The other changes were needed to run the package on its own, on Windows and with current packages:

- **Imports:** `marlbase.*` became `ac.*`, and `run.py` and `evaluate.py` add the repo root to `sys.path`. Scripts can then be run directly from this folder, including in the sub-processes that `AsyncVectorEnv` starts on Windows.
- **Launchers:** `run.py` and `evaluate.py` come from `marlbase/run.py` and `marlbase/eval.py`. The evaluation launcher is renamed to avoid clashing with `ac/eval.py`.
- **Environments:** SMAClite support was removed from `utils/envs.py`, and the single-env and parallel-env builders now share one function.
- **Output folders:** a `path_safe` resolver removes `:` from environment names. The multirun sweep folder no longer uses `algorithm.name`, which Hydra cannot resolve at sweep level.
- **Launcher plugin:** the default `submitit_local` launcher was dropped, so multiruns use Hydra's built-in sequential launcher.
- **Video:** `VideoRecorder` falls back to `render(mode="rgb_array")` for lbforaging 2.x, which ignores `render_mode`.
- **Requirements:**
  - `torch` 2.4.0 → 2.4.1, because 2.4.0 fails to load `fbgemm.dll` on Windows.
  - `hydra-ax-sweeper` removed, because it crashes Hydra's plugin loading on Python 3.11+.
  - `lbforaging==2.0.0` added.

## References

- Albrecht, S. V., Christianos, F., & Schäfer, L. (2024). *Multi-Agent Reinforcement Learning: Foundations and Modern Approaches.* MIT Press. [marl-book.com](https://www.marl-book.com)
- Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML.*
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347.*
- Yu, C., et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. *NeurIPS Datasets and Benchmarks.*
- Christianos, F., et al. (2021). Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing. *ICML.*
- Papoudakis, G., et al. (2021). Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks. *NeurIPS Datasets and Benchmarks.*
