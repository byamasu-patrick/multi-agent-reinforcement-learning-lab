# Multi-Agent Reinforcement Learning Lab

A hands-on exploration of Multi-Agent Reinforcement Learning (MARL) through implementations built from first principles. This repository documents my journey toward understanding how multiple intelligent agents learn, communicate, coordinate, and compete in shared environments.

The goal is to develop a deep understanding of the algorithms, communication mechanisms, and learning paradigms that underpin modern multi-agent systems.

## Implemented so far

| Folder | Algorithms | Setting | Environment | Runs on |
|---|---|---|---|---|
| [`tabular_marl/`](tabular_marl/) | Independent Q-Learning (IQL) | Tabular, 2 agents | Prisoner's Dilemma (matrix game) | Local |
| [`ac/`](ac/) | IA2C, MAA2C, IPPO, MAPPO | Deep, $N$ agents | Level-Based Foraging | Local or [Hugging Face Jobs](#running-experiments-on-hugging-face-jobs) |

Each folder is self-contained, with its own `requirements.txt`, virtual environment and README. Each README covers the algorithm's maths, how to run it and its results.

### Tabular: Independent Q-Learning

[`tabular_marl/`](tabular_marl/README.md): two agents learn the Prisoner's Dilemma with independent ε-greedy Q-learning. Both converge to mutual defection, the Nash equilibrium. The Q-value curves show the non-stationarity of independent learning: each agent's Q-values follow the other agent's changing exploration rate, and they match a closed-form prediction.

### Deep actor-critic: IA2C, MAA2C, IPPO, MAPPO

[`ac/`](ac/README.md): four actor-critic algorithms sharing one PyTorch implementation, configured with Hydra. They differ along two axes:

|  | Decentralised critic $V(o_i)$ | Centralised critic $V(o_1, \dots, o_N)$ |
|---|---|---|
| **A2C update** | IA2C | MAA2C |
| **PPO update** | IPPO | MAPPO |

The centralised variants are centralised training with decentralised execution (CTDE): only the critic sees every agent's observation, and each actor still acts from its own. The code started as the `ac` module of the [MARL book codebase](https://github.com/marl-book/codebase), and has been separated from it to run on its own.

## Running experiments

Each implementation has its own environment, because their dependencies differ. `ac` needs Python ≤ 3.12 for `torch==2.4.1`; `tabular_marl` runs on any recent Python.

```powershell
cd tabular_marl
py -3.12 -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
python train_iql.py
```

```powershell
cd ac
py -3.12 -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
python run.py +algorithm=mappo env.name="lbforaging:Foraging-8x8-2p-3f-v3" env.time_limit=25
```

### Running experiments on Hugging Face Jobs

Experiments in `ac/` can run on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) by adding `--hf-job` to the same command:

```powershell
python run.py +algorithm=mappo env.name="lbforaging:Foraging-8x8-2p-3f-v3" env.time_limit=25 --hf-job
```

This uses [`hf-jobs-launch`](https://github.com/byamasu-patrick/rl-algorithms-lab/tree/master/hf-training-jobs), the launcher shared with my single-agent CleanRL implementations in [rl-algorithms-lab](https://github.com/byamasu-patrick/rl-algorithms-lab). It:

1. uploads the code to a private Hugging Face bucket,
2. runs the experiment on the chosen hardware,
3. copies Hydra's `outputs/` and `multirun/` folders back to the bucket.

A Hugging Face account with Jobs enabled and `hf auth login` are required. [`ac/README.md`](ac/README.md#running-on-hugging-face-jobs) covers the hardware defaults, sweeps, and downloading results.

## Repository structure

```text
multi-agent-reinforcement-learning-lab/
├── tabular_marl/          # tabular MARL: independent Q-learning on matrix games
│   ├── iql.py             # IQL agents: ε-greedy action selection and Q-learning updates
│   ├── matrix_game.py     # two-player normal-form game environment (Prisoner's Dilemma)
│   ├── train_iql.py       # training, evaluation and plots
│   ├── utils.py
│   ├── figures/           # saved results
│   └── README.md
├── ac/                    # deep multi-agent actor-critic: IA2C, MAA2C, IPPO, MAPPO
│   ├── run.py             # training entry point (Hydra; --hf-job for Hugging Face Jobs)
│   ├── evaluate.py        # record a video of a trained checkpoint
│   ├── train.py           # trajectory collection and training loop
│   ├── model.py           # A2C and PPO networks and losses
│   ├── configs/           # Hydra configs: defaults, algorithms, loggers
│   ├── utils/             # environments, wrappers, networks, returns, logging
│   ├── pyproject.toml     # Hugging Face Jobs settings
│   └── README.md
└── README.md
```

## Topics

### Foundations

* Markov Games (Stochastic Games)
* Decentralized and Centralized Learning
* Cooperative, Competitive, and Mixed Settings
* Partial Observability
* Credit Assignment
* Multi-Agent Coordination
* Communication Learning
* Value Decomposition
* Centralized Training with Decentralized Execution (CTDE)

## Roadmap

✅ implemented · ⬜ planned

### Independent Learning

* ✅ Independent Q-Learning (IQL)
* ✅ Independent Advantage Actor-Critic (IA2C)
* ✅ Independent PPO (IPPO)
* ⬜ Independent Deep Q-Network (IDQN)

### Value-Based Methods

* ⬜ Value Decomposition Networks (VDN)
* ⬜ QMIX
* ⬜ QTRAN
* ⬜ QPLEX

### Actor-Critic Methods

* ✅ MAA2C
* ✅ MAPPO
* ⬜ MADDPG
* ⬜ COMA
* ⬜ HATRPO
* ⬜ HAPPO

### Communication Learning

* ⬜ CommNet
* ⬜ TarMAC
* ⬜ IC3Net
* ⬜ DIAL

### Graph-Based Multi-Agent Learning

* ⬜ Graph Neural Network-based MARL
* ⬜ Graph Attention for Multi-Agent Communication
* ⬜ Dynamic Interaction Graphs
* ⬜ Resource-Efficient Communication

### Advanced Topics

* ⬜ Curriculum Learning
* ⬜ Population-Based Training
* ⬜ Emergent Communication
* ⬜ Mean Field Reinforcement Learning
* ⬜ Large-Scale Multi-Agent Systems

## Motivation

Many real-world problems involve multiple autonomous agents interacting within dynamic environments. These interactions require coordination, communication, cooperation, or competition under uncertainty.

This repository serves as both a learning resource and a research platform for implementing, understanding, and evaluating modern Multi-Agent Reinforcement Learning algorithms from the ground up.

## Technologies

* Python
* PyTorch
* NumPy
* Gymnasium
* Hydra
* Matplotlib
* Hugging Face Jobs, via [`hf-jobs-launch`](https://github.com/byamasu-patrick/rl-algorithms-lab/tree/master/hf-training-jobs)

## Long-Term Goals

* Build MARL algorithms from first principles.
* Explore scalable coordination and communication strategies.
* Investigate Graph Neural Networks for Multi-Agent Reinforcement Learning.
* Study efficient communication under partial observability.
* Reproduce influential MARL research papers.
* Benchmark algorithms across standard multi-agent environments.

## References

* Albrecht, S. V., Christianos, F., & Schäfer, L. (2024). Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. MIT Press.
* Littman, M. L. (1994). Markov Games as a Framework for Multi-Agent Reinforcement Learning.
* Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
* Rashid, T., et al. (2018). QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.
* Sunehag, P., et al. (2018). Value-Decomposition Networks for Cooperative Multi-Agent Learning.
* Papoudakis, G., et al. (2021). Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks.
* Yu, C., et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
* Jiang, J., & Lu, Z. (2020). Learning Attentional Communication for Multi-Agent Cooperation.
