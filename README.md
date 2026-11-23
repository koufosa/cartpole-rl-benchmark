# CartPole RL Benchmark

**Random control vs. a physics heuristic vs. a learned neural-network policy** on Gymnasium's `CartPole-v1`.

> **Project status:** Official Gymnasium benchmark completed successfully on November 23, 2026. See the [verified results report](reports/official-gymnasium-benchmark.md) and [successful GitHub Actions run](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740).

## Research question

Can a learned control policy balance the pole more reliably than random actions and a simple angle-based controller? How does exploration change learning?

## What this project demonstrates

- Sequential decision-making in the CartPole environment (state = cart position, cart velocity, pole angle, pole angular velocity; actions = left or right).
- Three controller families: random baseline, hand-crafted physical heuristic, and a neural-network policy.
- **Elite-episode cross-entropy method (CEM):** collect trajectories, select top-performing episodes, and use their state/action pairs to update an `MLPClassifier` with `partial_fit`.
- ε-greedy **exploration**: with probability ε, choose a random action; otherwise sample from the current policy. ε decays across training epochs.
- Evaluation with held-out, matching environment seeds across agents; 100 episodes by default.
- Training plots, evaluation boxplots, and per-episode CSV exports.

**Algorithm clarification:** This project uses CEM-style supervised policy search, **not Q-learning, deep Q-network (DQN), or policy-gradient REINFORCE**. The neural network predicts actions, not action-value estimates. The environment's episode returns are used to *select* training trajectories, not as a gradient target.

## Installation

Requirements: Python 3.10+, a terminal, and optionally a graphical desktop for the render command.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
```

## Reproduce the experiment

```bash
# Train the learned policy (random and heuristic agents require no training)
cartpole-benchmark train --epochs 100 --episodes-per-epoch 40 --seed 42

# Evaluate the saved agent and both baselines on the same 100 environment seeds
cartpole-benchmark evaluate --model outputs/policy.joblib --episodes 100 --seed 10000

# Watch a trained policy in a graphical environment
cartpole-benchmark demo --agent neural --model outputs/policy.joblib

# Tests
python -m pytest -q
```

For a fast smoke test, use `cartpole-benchmark train --epochs 2 --episodes-per-epoch 4` and `cartpole-benchmark evaluate --episodes 5`. Those short runs do **not** establish meaningful performance.

## Repository structure

```text
cartpole-rl-benchmark/
├── .github/workflows/tests.yml   # CI test workflow
├── src/cartpole_benchmark/
│   ├── agents.py                  # Controllers and neural policy
│   ├── env.py                     # Gymnasium episode runner
│   ├── training.py                # Elite-episode training and CSV logs
│   ├── evaluation.py              # Matched-seed benchmark
│   ├── plots.py                   # Training/evaluation visualization
│   └── cli.py                     # Command line entry point
├── tests/                         # Agent and episode unit tests
├── outputs/                       # Generated locally; not committed
├── pyproject.toml
└── README.md
```

## Evaluation protocol

- `CartPole-v1` runs up to 500 timesteps. Its reward is +1 per timestep survived.
- All three agents are evaluated on *the same set of reset seeds*. Training uses different seeds than evaluation.
- The neural agent is evaluated greedily (argmax action probability, ε=0), rather than with exploratory action sampling.
- Report mean, standard deviation, median, minimum, maximum, and the share of evaluation episodes achieving **at least 475** reward.
- The random baseline has a seeded RNG, though its action sequence depends on previous episode lengths. Exact action-level pairing across agents is not attempted.
- Training reward uses exploratory trajectories, so it is **not** directly interchangeable with deterministic test reward.

### Verified official Gymnasium results

[GitHub Actions run #2](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740) installed Gymnasium, ran tests, trained for 100 epochs (40 episodes each) and evaluated all agents over the same 100 held-out environment seeds.

| Controller | Mean reward (100 episodes) | Std. dev. | Success rate ≥475 |
| --- | ---: | ---: | ---: |
| Random | 21.7 | 11.0 | 0% |
| Physics heuristic | 500.0 | 0.0 | 100% |
| Neural policy | 467.1 | 63.1 | 76% |

The neural policy substantially outperformed random control, but the handcrafted physics heuristic remained stronger in the tested environment. All results are from **one trained policy** across 100 evaluation episodes; multiple training seeds would support stronger generalization claims.

Read the [official reproducibility report](reports/official-gymnasium-benchmark.md), or [download the official CSVs, plots, and policy](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740/artifacts/14852625271). The older [provisional local physics reproduction](reports/provisional-benchmark.md) is retained separately for transparency.

**Limitation:** CEM can plateau, and a basic MLP policy is not guaranteed to solve CartPole. Compare multiple training seeds, percentile thresholds, and epsilon schedules before making strong conclusions.


## Tools

Python, Gymnasium, NumPy, scikit-learn, Matplotlib, pytest, GitHub Actions.


