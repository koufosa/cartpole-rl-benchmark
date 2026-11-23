"""Gymnasium environment helpers."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Episode:
    reward: float
    states: list[np.ndarray]
    actions: list[int]


def make_env(render_mode: str | None = None):
    import gymnasium as gym
    return gym.make("CartPole-v1", render_mode=render_mode)


def play_episode(env, agent, seed: int, epsilon: float = 0.0, greedy: bool = False) -> Episode:
    state, _ = env.reset(seed=seed)
    states, actions = [], []
    total_reward = 0.0
    done = False
    while not done:
        action = agent.greedy_action(state) if greedy else agent.act(state, epsilon=epsilon)
        states.append(np.asarray(state, dtype=float).copy())
        actions.append(int(action))
        state, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        done = bool(terminated or truncated)
    return Episode(total_reward, states, actions)
