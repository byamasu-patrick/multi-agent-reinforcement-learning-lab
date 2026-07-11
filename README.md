# Multi-Agent Reinforcement Learning Lab

A hands-on exploration of Multi-Agent Reinforcement Learning (MARL) through implementations built from first principles. This repository documents my journey toward understanding how multiple intelligent agents learn, communicate, coordinate, and compete in shared environments.

The goal is to develop a deep understanding of the algorithms, communication mechanisms, and learning paradigms that underpin modern multi-agent systems.

## Topics Covered

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

## Implemented Algorithms

### Independent Learning

* Independent Q-Learning (IQL)
* Independent Deep Q-Network (IDQN)

### Value-Based Methods

* Value Decomposition Networks (VDN)
* QMIX
* QTRAN
* QPLEX

### Actor-Critic Methods

* MADDPG
* COMA
* MAPPO
* HATRPO
* HAPPO

### Communication Learning

* CommNet
* TarMAC
* IC3Net
* DIAL

### Graph-Based Multi-Agent Learning

* Graph Neural Network-based MARL
* Graph Attention for Multi-Agent Communication
* Dynamic Interaction Graphs
* Resource-Efficient Communication

### Advanced Topics

* Curriculum Learning
* Population-Based Training
* Emergent Communication
* Mean Field Reinforcement Learning
* Large-Scale Multi-Agent Systems

## Motivation

Many real-world problems involve multiple autonomous agents interacting within dynamic environments. These interactions require coordination, communication, cooperation, or competition under uncertainty.

This repository serves as both a learning resource and a research platform for implementing, understanding, and evaluating modern Multi-Agent Reinforcement Learning algorithms from the ground up.

## Repository Structure

```text
multi-agent-reinforcement-learning-lab/
│
├── notebooks/
│   ├── markov_games.ipynb
│   ├── independent_q_learning.ipynb
│   ├── vdn.ipynb
│   ├── qmix.ipynb
│   ├── maddpg.ipynb
│   ├── mappo.ipynb
│   └── gnn_marl.ipynb
│
├── src/
│   ├── agents/
│   ├── networks/
│   ├── environments/
│   ├── communication/
│   └── utils/
│
├── experiments/
│
├── environments/
│
├── datasets/
│
├── README.md
└── requirements.txt
```

## Technologies

* Python
* PyTorch
* NumPy
* Gymnasium
* PettingZoo
* NetworkX
* Matplotlib

## Long-Term Goals

* Build MARL algorithms from first principles.
* Explore scalable coordination and communication strategies.
* Investigate Graph Neural Networks for Multi-Agent Reinforcement Learning.
* Study efficient communication under partial observability.
* Reproduce influential MARL research papers.
* Benchmark algorithms across standard multi-agent environments.

## References

* Littman, M. L. (1994). Markov Games as a Framework for Multi-Agent Reinforcement Learning.
* Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.
* Rashid, T., et al. (2018). QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.
* Sunehag, P., et al. (2018). Value-Decomposition Networks for Cooperative Multi-Agent Learning.
* Yu, C., et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.
* Jiang, J., & Lu, Z. (2020). Learning Attentional Communication for Multi-Agent Cooperation.
