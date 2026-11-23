"""Train neural policy using elite-episode cross-entropy method."""
from __future__ import annotations
import csv
from pathlib import Path
import joblib
import numpy as np
from .agents import NeuralPolicyAgent
from .env import make_env, play_episode


def train(output_dir: str | Path, *, epochs: int = 100, episodes_per_epoch: int = 40,
          percentile: float = 70, epsilon_start: float = 0.9,
          epsilon_end: float = 0.05, seed: int = 42) -> tuple[NeuralPolicyAgent, list[dict]]:
    if epochs < 1 or episodes_per_epoch < 2:
        raise ValueError("epochs >= 1 and episodes_per_epoch >= 2 required")
    if not 0 <= percentile < 100:
        raise ValueError("percentile must be in [0, 100)")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = NeuralPolicyAgent(seed=seed)
    history: list[dict] = []
    env = make_env()
    try:
        for epoch in range(epochs):
            fraction = epoch / max(epochs - 1, 1)
            epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
            batch = [play_episode(env, agent, seed + epoch * episodes_per_epoch + i, epsilon)
                     for i in range(episodes_per_epoch)]
            rewards = np.asarray([ep.reward for ep in batch], dtype=float)
            cutoff = float(np.percentile(rewards, percentile))
            elite = [ep for ep in batch if ep.reward >= cutoff]
            states = np.vstack([ep.states for ep in elite])
            actions = np.asarray([a for ep in elite for a in ep.actions])
            agent.update(states, actions)
            record = {"epoch": epoch + 1, "epsilon": round(epsilon, 4),
                      "mean_reward": round(float(np.mean(rewards)), 3),
                      "median_reward": round(float(np.median(rewards)), 3),
                      "max_reward": round(float(np.max(rewards)), 3),
                      "elite_episodes": len(elite)}
            history.append(record)
            print(f"Epoch {epoch+1:3d}/{epochs} | mean={record['mean_reward']:6.1f} "
                  f"| best={record['max_reward']:3.0f} | epsilon={epsilon:.3f}")
    finally:
        env.close()
    with (output_dir / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    joblib.dump(agent, output_dir / "policy.joblib")
    return agent, history
