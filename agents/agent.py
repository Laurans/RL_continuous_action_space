import numpy as np
from utils.replaybuffer import ReplayBuffer
from actors.actor import Actor
from critics.critic import Critic
from agents.noise import OUNoise

class Agent():
    def __init__(self, cfg):
        """
        """

        # Replay memory
        self.memory = ReplayBuffer(**cfg['agent']['memory'])

        # Environment configuration
        self.state_shape = self.memory.window_state_shape
        self.action_shape = cfg['env']['action_shape']
        self.action_low, self.action_high = cfg['env']['action_range']


        # Actor Model
        self.actor_local = Actor(self.state_shape, self.action_shape, self.action_low, self.action_high, cfg['actor'])
        self.actor_target = Actor(self.state_shape, self.action_shape, self.action_low, self.action_high, cfg['actor'])

        # Critic Model
        self.critic_local = Critic(self.state_shape, self.action_shape, cfg['critic'])
        self.critic_target = Critic(self.state_shape, self.action_shape, cfg['critic'])

        # Initialize target model
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Algorithm parameters
        self.exploration_mu, self.exploration_theta, self.exploration_sigma = cfg['agent']['noise']
        self.noise = OUNoise(self.action_shape, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        self.gamma = cfg['agent']['gamma']
        self.tau = cfg['agent']['tau']

    def _reshape_state(self, state):
        return np.reshape(state, [-1, self.state_shape])

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward,
                        next_state, done)

        # Learn, if at end of episode
        if done and self.memory.is_sufficient():
            self.learn()
            self.noise = OUNoise(self.action_shape, self.exploration_mu, self.exploration_theta, self.eps)

    def act(self, state):
        self.last_state = state

        window_states = self.memory.get_last_past_states(state)
        action = self.actor_local.model.predict(window_states) + self.noise.sample()


        return action

    def learn(self):
        experiences = self.memory.sample()

        states = experiences['state'][:, 0]
        actions = experiences['action']
        rewards = experiences['reward']
        dones = experiences['done']
        next_states = experiences['next_state'][:, 0]

        # get predicted next state action and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        loss1 = self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model
        action_gradients = self.critic_local.get_action_gradients([states, actions, 0])[0]
        ##print('DEBUG', action_gradients)
        loss2 = self.actor_local.train_fn([states, action_gradients, 1])[0]
       
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