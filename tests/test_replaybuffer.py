import unittest
from utils.replaybuffer import ReplayBuffer
import numpy as np


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.cls = ReplayBuffer(buffer_size=1000,
                                batch_size=4,
                                past_size=2)

        states = np.arange(50).reshape(-1, 2)
        next_states = np.arange(2, 52).reshape(-1, 2)
        actions = np.arange(0, 25, 1) ** 2
        rewards = np.linspace(-1, 2, 25)
        terms = np.zeros(25)

        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terms):
            self.cls.add(s, a, r, s2, d)

    def test_lenght_memory(self):
        self.assertEqual(25, len(self.cls))

    def test_memory(self):
        self.assertTrue(np.array([6, 7]).all() == self.cls.memory[3]['state'].all())

    def test_get_historic_state(self):
        self.assertTrue(
            np.array([[0, 1], [2, 3], [4, 5]]).all() == \
            self.cls.get(2)['state'].all())

    def test_sample_size(self):
        self.assertEqual(4, len(self.cls.sample()))


class TestMemoryOverwrite(unittest.TestCase):
    def setUp(self):
        self.cls = ReplayBuffer(24, 4, 2)

        states = np.arange(50).reshape(-1, 2)
        next_states = np.arange(2, 52).reshape(-1, 2)
        actions = np.arange(0, 25, 1) ** 2
        rewards = np.linspace(-1, 2, 25)
        terms = np.zeros(25)

        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terms):
            self.cls.add(s, a, r, s2, d)

    def test_rolling(self):
        self.assertTrue(np.all(np.array([2, 3]) == self.cls.memory[0]['state']))

    def test_overwritting(self):
        self.assertTrue(np.all(np.array([48, 49]) == self.cls.memory[-1]['state']))

#if __name__ == '__main__':
#    unittest.main()
