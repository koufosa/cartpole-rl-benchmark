"""Save training and evaluation figures to PNG."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training(history: list[dict], path: str | Path):
    epochs = [row["epoch"] for row in history]
    means = [row["mean_reward"] for row in history]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(epochs, means, label="Training batch mean reward")
    ax.axhline(475, linestyle="--", color="gray", label="475 / 500 benchmark")
    ax.set(xlabel="Epoch", ylabel="Average total reward", title="Neural policy training")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_evaluation(details: list[dict], path: str | Path):
    names = list(dict.fromkeys(row["agent"] for row in details))
    groups = [[row["reward"] for row in details if row["agent"] == name] for name in names]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.boxplot(groups, tick_labels=names, showmeans=True)
    ax.set(ylabel="Episode reward", title="Evaluation on matched episode seeds")
    ax.set_ylim(0, 510)
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
