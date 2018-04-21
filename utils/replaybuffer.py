import numpy as np
import sys
from collections import deque

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, past_size, state_shape):
        """
        Initialize a ReplayBuffer object.
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param past_size: number of past experiences to pack with the current experience for training
        """
        self.max_size = buffer_size
        self.past_size = past_size
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.flatten_state_shape = np.prod([past_size + 1] + state_shape)

        self.experience_type = [('state', (np.float32, self.flatten_state_shape)),
                                ('action', np.float32),
                                ('reward', np.float32),
                                ('next_state', (np.float32, self.flatten_state_shape)),
                                ('done', np.int32)]

        self.past = deque(maxlen=past_size)
        self.reset_past()

        self.cursor_insert = 0
        self.cursor_full = 0

        self.memory = np.zeros(buffer_size, dtype=self.experience_type)

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

        self.cursor_insert = (self.cursor_insert + 1) % self.max_size

        if self.cursor_full < self.max_size:
            self.cursor_full += 1

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.choice(np.arange(self.cursor_full), size=self.batch_size, replace=False)

        return np.vstack(self.memory[indices])

    def __len__(self):
        """Return the current size of internal memory."""
        return self.cursor_full

    def is_sufficient(self):
        """Return True if we can start sampling"""
        return len(self.memory) > self.batch_size
