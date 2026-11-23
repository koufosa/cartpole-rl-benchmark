import numpy as np
import pytest
from cartpole_benchmark.agents import PhysicsAgent, RandomAgent, NeuralPolicyAgent


def test_physics_responds_to_angle():
    agent = PhysicsAgent()
    assert agent.act(np.array([0, 0, 0.1, 0])) == 1
    assert agent.act(np.array([0, 0, -0.1, 0])) == 0


def test_random_agent_actions_valid():
    agent = RandomAgent(seed=12)
    assert {agent.act(np.zeros(4)) for _ in range(100)} == {0, 1}


def test_neural_policy_fit_and_action():
    agent = NeuralPolicyAgent(seed=42)
    states = np.array([[0, 0, -0.1, -0.1], [0, 0, 0.1, 0.1],
                       [0, 0, -0.2, -0.1], [0, 0, 0.2, 0.1]])
    agent.update(states, np.array([0, 1, 0, 1]))
    assert agent.greedy_action(states[0]) in (0, 1)
    assert agent.act(states[1], epsilon=0.5) in (0, 1)


def test_epsilon_range():
    with pytest.raises(ValueError):
        NeuralPolicyAgent().act(np.zeros(4), epsilon=1.1)
