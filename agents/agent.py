import numpy as np


class Agent():
    def __init__(self, cfg):
        """

        :param state_size: size of the observation vector received by the agent
        :param action_size: number of actions available to the agent
        :param eps: proportion to explore more states
        :param alpha: learning rate
        :param gamma: discount factor
        """
        self.state_size = cfg['env']['state_size']
        self.action_size = cfg['env']['action_size']
        self.action_low = cfg['env']['action_low']
        self.action_high = cfg['env']['action_high']
        self.eps = cfg['agent']['epsilon']
        self.alpha = cfg['agent']['alpha']
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