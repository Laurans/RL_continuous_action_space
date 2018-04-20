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

        self.action_shape = cfg['env']['action_shape']

        # Algorithm parameters
        self.exploration_mu, self.exploration_theta, self.exploration_sigma = cfg['agent']['noise']
        self.noise = OUNoise(self.action_shape, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        self.gamma = cfg['agent']['gamma']
        self.tau = cfg['agent']['tau']

        state_flatten_shape = [np.prod(self.memory.window_state_shape)]
        # Actor Model
        self.actor = Actor(state_flatten_shape, self.action_shape, cfg['env']['action_range'],
                           self.tau, self.memory.batch_size, cfg['actor'])

        # Critic Model
        self.critic = Critic(state_flatten_shape, self.action_shape, self.tau, cfg['critic'])

    def init_actor_critic(self):
        # Initialize target model
        self.critic.copy_local_in_target()
        self.actor.copy_local_in_target()


    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward,
                        next_state, done)

        # Learn, if at end of episode
        if done and self.memory.is_sufficient():
            return self.learn()

    def act(self, state):
        self.last_state = state

        window_states = self.memory.get_last_past_states(state).reshape(1, -1)
        action = self.actor.predict(window_states) + self.noise.sample()

        return np.clip(action, -1, 1)

    def learn(self):
        experiences = self.memory.sample()

        states = experiences['state'][:, 0].reshape(self.memory.batch_size, -1)
        actions = experiences['action']
        rewards = experiences['reward']
        dones = experiences['done']
        next_states = experiences['next_state'][:, 0].reshape(self.memory.batch_size, -1)


        # get predicted next state action and Q values from target models
        actions_next = self.actor.get_targets(next_states)
        Q_targets_next = self.critic.get_targets(next_states, actions_next)

        # Compute Q targets for current states and train critic model
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        summary = self.critic.fit(states, actions, Q_targets)

        # Train actor model
        action_gradients = self.critic.get_actions_grad(states, actions)[0]
        self.actor.fit(states, action_gradients)

        # Soft-update target models
        self.critic.soft_update()
        self.actor.soft_update()
        return summary