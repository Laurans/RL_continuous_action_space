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
        self.state_size = cfg['env']['state_size']
        self.action_size = cfg['env']['action_size']
        self.action_low, self.action_high = cfg['env']['action_range']

        # Actor Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, cfg['actor'])
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, cfg['actor'])

        # Critic Model
        self.critic_local = Critic(self.state_size, self.action_size, cfg['critic'])
        self.critic_target = Critic(self.state_size, self.action_size, cfg['critic'])

        # Initialize target model

        # Replay memory
        self.memory = ReplayBuffer(**cfg['agent']['memory'])

        # Algorithm parameters
        self.eps, self.eps_min, self.eps_decay = cfg['agent']['explore']
        self.gamma = cfg['agent']['gamma']

        self.i_episode = 0

    def step(self, reward, done):
        # Save experience / reward

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        action = (self.action_high - self.action_low) * \
                 np.random.random_sample() + self.action_low
        return action

    def learn(self):
        pass