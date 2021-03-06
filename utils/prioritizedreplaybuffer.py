import numpy as np
from collections import deque
from utils.sumtree import SumTree

class PrioritizedReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, past_size, state_shape, alpha, epsilon):
        """
        Initialize a ReplayBuffer object.
        """
        self.max_size = buffer_size
        self.past_size = past_size
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.flatten_state_shape = np.prod([past_size + 1] + state_shape)
        self.alpha = alpha
        self.epsilon = epsilon

        self.experience_type = [('state', (np.float32, self.flatten_state_shape)),
                                      ('action', np.float32),
                                      ('reward', np.float32),
                                      ('next_state', (np.float32, self.flatten_state_shape)),
                                      ('done', np.int32)]

        self.past = deque(maxlen=past_size)
        self.reset_past()

        self.cursor_insert = 0
        self.cursor_size = 0

        self.memory = np.zeros(buffer_size, dtype=self.experience_type)
        self.sumtree = SumTree(buffer_size)

    def reset_past(self):
        self.past.extend([np.zeros(self.state_shape)] * self.past_size)

    def get_state_vector(self, state):
        past = list(self.past)
        past.append(state.copy().reshape(self.state_shape))
        return np.vstack(past).ravel()

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        # Get past
        experience_state = self.get_state_vector(state)

        # Add state
        self.past.append(state.copy().reshape(self.state_shape))

        # Get past of the next state
        experience_next_state = self.get_state_vector(next_state)
        self.memory[self.cursor_insert] = (experience_state, action, reward, experience_next_state, done)
        self.sumtree.add(self.cursor_insert, 1)

        self.cursor_insert = (self.cursor_insert+1) % self.max_size

        if self.cursor_size < self.max_size:
            self.cursor_size += 1

    def sample(self):
        references = []
        self.last_prio_indices = []
        weights = []

        for _ in range(self.batch_size):
            #value = np.random.random() * self.sumtree.total()
            ref, prio, idx = self.sumtree.find(np.random.random())
            references.append(ref)
            self.last_prio_indices.append(idx)
            proba = prio / self.sumtree.total()
            w = (self.cursor_size * proba) ** (-1) if prio > 1e-16 else 0
            weights.append(w)

        #import ipdb; ipdb.set_trace()
        weights = weights / max(weights)

        return np.vstack(self.memory[references]), np.array(weights).reshape((self.batch_size, 1))

    def update_priority(self, td_errors):
        for error, indice in zip(td_errors, self.last_prio_indices):
            prio = (np.abs(error)+self.epsilon)**self.alpha
            self.sumtree.val_update(indice, prio)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.cursor_size

    def is_sufficient(self):
        """Return True if we can start sampling"""
        return len(self.memory) > self.batch_size