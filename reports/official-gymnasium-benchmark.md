# Official Gymnasium benchmark results

**Verified run:** [CartPole experiment #2](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740) · commit `830a772a5a6d305ac9274b9fa4e1c9fc9c2485dc` · November 23, 2026 · **success**

This is a **real Gymnasium CartPole-v1 execution**, unlike the earlier provisional local physics reproduction. The run installed dependencies, executed tests and training, evaluated the three agents, and published the six output files in a GitHub Actions artifact.

## Experimental protocol

- Environment: Gymnasium `CartPole-v1`, maximum episode reward 500
- Model: scikit-learn MLPClassifier with hidden layers (64, 64)
- Training: elite-episode cross-entropy method, **not DQN**
- Training episodes: 100 epochs × 40 episodes per epoch = 4,000 episodes
- Training reset seed: 42, incremented for each episode
- Exploration: linearly decaying ε from 0.9 to 0.05
- Evaluation: 100 held-out environment seeds, 10000–10099
- Learned model evaluated deterministically (greedy policy)

## Official results

| Controller | Mean reward | Standard deviation | Episodes scoring at least 475 |
| --- | ---: | ---: | ---: |
| Random | 21.7 | 11.0 | 0 / 100 |
| Physics heuristic | 500.0 | 0.0 | 100 / 100 |
| Neural network | 467.1 | 63.1 | 76 / 100 |

**Interpretation.** The learned neural policy achieved **467.1 / 500 average reward** after training, substantially improving upon random actions, but did **not** outperform the handcrafted physics heuristic. The heuristic solved all 100 evaluation episodes under the tested seeds.

The training output shows exploratory average reward grew from low initial values to **108.3** at epoch 100. This differs from the 467.1 evaluation average because training episodes use ε-greedy exploration, whereas evaluation uses deterministic greedy actions. These figures must not be confused.

## Generated artifacts

[Download the official 6-file run artifact](https://github.com/koufosa/cartpole-rl-benchmark/actions/runs/45241257740/artifacts/14852625271):

- `training_history.csv`
- `evaluation_episodes.csv`
- `evaluation_summary.csv`
- `training_curve.png`
- `evaluation_boxplot.png`
- `policy.joblib`

The experiment was confirmed in the workflow logs. Numerical results above are rounded from stdout. The CSVs and PNGs are available in the artifact; they are not part of the repository tree.

## Limitations and improvements

- Results are from **one model training seed**, with 100 evaluation episodes, not multiple independent training trials.
- A simpler heuristic was more effective on this environment; future work should analyze why and test harder control tasks.
- Perform repeated independent training runs, tune exploration/elite selection parameters, and compare reward distributions.
- Confirm reproducibility across platforms and software versions.

The earlier [provisional comparison](provisional-benchmark.md) is kept for provenance, clearly marked as a local physics reproduction.
