import numpy as np

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, past_size):
        """
        Initialize a ReplayBuffer object.
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param past_size: number of past experiences to pack with the current experience for training
        """
        self.past_size = past_size
        self.max_size = buffer_size
        self.batch_size = batch_size

        self.insert_index = 0

        experience_type = [('state', (np.float32, 2)), ('action', np.float32), ('reward', np.float32),
                           ('next_state', (np.float32, 2)), ('done', np.int32)]

        self.batch_experience_type = [('state', (np.float32, (past_size+1,2))),
                                      ('action', np.float32),
                                      ('reward', np.float32),
                                      ('next_state', (np.float32, (past_size+1,2))),
                                      ('done', np.int32)]

        self.memory = np.zeros(buffer_size, dtype=experience_type)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        assert state is not None, 'State cannot be None'
        assert action is not None, 'Action cannot be None'
        assert reward is not None, 'Reward cannot be None'
        assert next_state is not None, 'State 2 cannot be None'

        experience = (state.copy(), action, reward, next_state.copy(), done)

        if self.insert_index == self.max_size:
            # Roll the memory to overwrite the oldest experience
            self.memory = np.roll(self.memory, -1, axis=0)
            index = -1
        else:
            index = self.insert_index

        self.memory[index] = experience

        # Increment until it is at full capacity
        if self.insert_index < self.max_size:
            self.insert_index += 1

    def get(self, index):
        """Get an experience with its past"""
        assert index < self.insert_index
        assert index >= self.past_size

        experience = self.memory[index - self.past_size: index + 1]

        new_experience = np.zeros(1, dtype=self.batch_experience_type)
        new_experience['state'] = np.vstack(experience['state'])
        new_experience['action'] = experience['action'][-1]
        new_experience['next_state'] = np.vstack(experience['next_state'])
        new_experience['reward'] = experience['reward'][-1]
        new_experience['done'] = experience['done'][-1]

        return new_experience

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indexes = np.random.randint(self.past_size, self.insert_index, self.batch_size)

        batch = np.vstack([self.get(i) for i in indexes])
        return batch

    def __len__(self):
        """Return the current size of internal memory."""
        return self.insert_index

    def is_sufficient(self):
        """Return True if we can start sampling"""
        return self.insert_index > self.batch_size + self.past_size