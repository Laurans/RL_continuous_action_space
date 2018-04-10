import numpy as np
from replaybuffer.replaybuffer import ReplayBuffer
from actors.actor import Actor
from critics.critic import Critic

class Agent():
    def __init__(self, cfg):
        """

        :param state_size: size of the observation vector received by the agent
        :param action_size: number of actions available to the agent
        :param eps: proportion to explore more states
        :param alpha: learning rate
        :param gamma: discount factor
        """

        # Environment configuration
        self.state_size = cfg['env']['state_size'][0]
        self.action_size = cfg['env']['action_size'][0]
        self.action_low, self.action_high = cfg['env']['action_range']


        # Actor Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, cfg['actor'])
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, cfg['actor'])

        # Critic Model
        self.critic_local = Critic(self.state_size, self.action_size, cfg['critic'])
        self.critic_target = Critic(self.state_size, self.action_size, cfg['critic'])

        # Initialize target model
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Replay memory
        self.memory = ReplayBuffer(**cfg['agent']['memory'])

        # Algorithm parameters
        self.eps, self.eps_min, self.eps_decay = cfg['agent']['explore']
        self.gamma = cfg['agent']['gamma']
        self.tau = cfg['agent']['tau']

    def _reshape_state(self, state):
        return np.reshape(state, [-1, self.state_size])

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward,
                        self._reshape_state(next_state), done)

        # Learn, if at end of episode
        if done and self.memory.is_sufficient():
            self.learn()
            self.eps = np.max([self.eps*self.eps_decay, self.eps_min])

    def act(self, state):
        self.last_state = self._reshape_state(state)

        if np.random.random() < self.eps:
            action = (self.action_high - self.action_low) * np.random.random_sample() + self.action_low
        else:
            action = self.actor_local.model.predict(self.last_state)
        return action

    def learn(self):
        experiences = self.memory.sample()

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # get predicted next state action and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """ Soft update model parameters"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights)

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)