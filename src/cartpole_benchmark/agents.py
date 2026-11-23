"""Baseline controllers and a learned neural-network policy."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.neural_network import MLPClassifier


class RandomAgent:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        return int(self.rng.integers(0, 2))


class PhysicsAgent:
    """Simple hand-crafted angle/angular-velocity controller, not an optimal controller."""
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        angle, angular_velocity = float(state[2]), float(state[3])
        return int(angle + 0.5 * angular_velocity > 0)


@dataclass
class NeuralPolicyAgent:
    """MLP policy trained on high-reward episodes (cross-entropy method).

    Important: this is episodic supervised policy search, not DQN or Q-learning.
    """
    seed: int = 42
    hidden: tuple[int, int] = (64, 64)
    learning_rate: float = 0.001

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            learning_rate_init=self.learning_rate,
            solver="adam",
            random_state=self.seed,
            max_iter=1,
        )
        self.fitted = False

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if not (0 <= epsilon <= 1):
            raise ValueError("epsilon must be between 0 and 1")
        if not self.fitted or self.rng.random() < epsilon:
            return int(self.rng.integers(0, 2))
        # Sampling from the learned policy adds stochasticity during training.
        prob_right = float(self.model.predict_proba(np.asarray(state).reshape(1, -1))[0, 1])
        return int(self.rng.random() < prob_right)

    def greedy_action(self, state: np.ndarray) -> int:
        if not self.fitted:
            raise RuntimeError("Train the policy before deterministic evaluation")
        probabilities = self.model.predict_proba(np.asarray(state).reshape(1, -1))[0]
        return int(self.model.classes_[int(np.argmax(probabilities))])

    def update(self, states: np.ndarray, actions: np.ndarray) -> None:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int)
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError("Expected states with shape (n, 4)")
        if len(states) != len(actions) or len(states) == 0:
            raise ValueError("states and actions must have matching nonzero lengths")
        self.model.partial_fit(states, actions, classes=np.array([0, 1]))
        self.fitted = True
