# CartPole Reinforcement Learning

I built this project to experiment with different ways of controlling the CartPole environment in Python. The goal is simple: move a cart left or right to keep a pole balanced for as long as possible.

What interested me was seeing how a model that learns from experience compares with a basic set of rules that already understands something about the physics.

## What I built

I compared three approaches:

1. **Random agent** — randomly moves left or right. This is my baseline.
2. **Physics-based agent** — uses the pole's angle and angular velocity to decide which way the cart should move.
3. **Neural network agent** — learns from higher-reward episodes, using a small neural network and epsilon-greedy exploration.

For the neural network, I used scikit-learn's `MLPClassifier` with two hidden layers of 64 neurons. It trains on the states and actions from the better episodes, using an elite-episode approach (often called the cross-entropy method). This isn't DQN or Q-learning; it's a simpler way to learn a policy from experience.

I also added scripts for training, evaluation, and plotting so I could compare the agents on the same test episodes.

## Results

I trained the neural network for **100 epochs**, with **40 episodes per epoch**, then evaluated all three approaches on **100 episodes** using Gymnasium's `CartPole-v1`. Each episode has a maximum reward of 500.

| Agent | Average reward | Episodes scoring at least 475 |
| --- | ---: | ---: |
| Random | 21.7 / 500 | 0 / 100 |
| Physics-based | 500.0 / 500 | 100 / 100 |
| Neural network | 467.1 / 500 | 76 / 100 |

The neural network was a major improvement over random actions, but the physics-based agent still performed better. I found that interesting because a well-chosen rule can outperform a learned model on a problem this simple.

The neural network also wasn't perfect every time. Its average score was high, but the results varied between episodes. I'd like to explore whether changing the training strategy or testing multiple random seeds makes the learned policy more consistent.

These numbers are from a completed [GitHub Actions run using Gymnasium](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740). You can find the [full results and charts here](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740/artifacts/14852625271).

## Running the project

You'll need Python 3.10 or newer.

```bash
git clone https://github.com/koufosa/cartpole-rl-benchmark.git
cd cartpole-rl-benchmark

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Train the neural network:

```bash
cartpole-benchmark train --epochs 100 --episodes-per-epoch 40
```

Compare the three agents:

```bash
cartpole-benchmark evaluate --episodes 100
```

Watch the trained agent balance the pole (requires a graphical display):

```bash
cartpole-benchmark demo --agent neural
```

Training and evaluation save CSV files and plots in the `outputs/` folder. Run `python -m pytest -q` to check the tests.

## What I used

Python, Gymnasium, NumPy, scikit-learn, Matplotlib, pytest, and GitHub Actions.

## What I want to try next

I'd like to compare different exploration rates, repeat training with several random seeds, and try a more advanced reinforcement learning approach to see if it can match or beat the physics-based controller.
