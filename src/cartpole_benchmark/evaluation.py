"""Seed-matched evaluation with summary statistics."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .agents import RandomAgent, PhysicsAgent
from .env import make_env, play_episode


def evaluate_agent(agent, seeds: list[int], *, greedy: bool = False) -> list[float]:
    env = make_env()
    try:
        return [play_episode(env, agent, seed=s, greedy=greedy).reward for s in seeds]
    finally:
        env.close()


def evaluate_all(policy, output_dir: str | Path, *, n_episodes: int = 100,
                 seed: int = 10000) -> tuple[list[dict], list[dict]]:
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(seed, seed + n_episodes))
    agents = {"random": (RandomAgent(seed=seed), False),
              "physics": (PhysicsAgent(), False),
              "neural_policy": (policy, True)}
    details, summary = [], []
    for name, (agent, greedy) in agents.items():
        rewards = evaluate_agent(agent, seeds, greedy=greedy)
        for s, reward in zip(seeds, rewards):
            details.append({"agent": name, "seed": s, "reward": reward})
        summary.append({"agent": name, "episodes": n_episodes,
                        "mean_reward": round(float(np.mean(rewards)), 3),
                        "std_reward": round(float(np.std(rewards)), 3),
                        "median_reward": round(float(np.median(rewards)), 3),
                        "min_reward": float(min(rewards)),
                        "max_reward": float(max(rewards)),
                        "success_rate_475": round(float(np.mean(np.asarray(rewards) >= 475)), 3)})
    for name, records in [("evaluation_episodes.csv", details), ("evaluation_summary.csv", summary)]:
        with (output_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary, details
