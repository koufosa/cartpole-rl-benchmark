import numpy as np
from cartpole_benchmark.env import play_episode


class FakeEnv:
    def reset(self, seed=None):
        self.steps = 0
        return np.zeros(4), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(4), 1.0, self.steps >= 3, False, {}


class AlwaysRight:
    def act(self, state, epsilon=0):
        return 1


def test_episode_records_actions_and_reward():
    episode = play_episode(FakeEnv(), AlwaysRight(), seed=42)
    assert episode.reward == 3.0
    assert len(episode.states) == 3
    assert episode.actions == [1, 1, 1]
