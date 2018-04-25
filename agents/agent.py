import numpy as np
from utils.replaybuffer import ReplayBuffer
from actors.actor import Actor
from critics.critic import Critic
from agents.noise import OUNoise
import tensorflow as tf

class Agent():
    def __init__(self, cfg):
        # Replay memory
        self.memory = ReplayBuffer(**cfg['agent']['memory'])

        # Environment configuration

        self.action_shape = cfg['env']['action_shape']

        # Algorithm parameters
        self.exploration_mu, self.exploration_theta, self.exploration_sigma = cfg['agent']['noise']
        self.noise = OUNoise(self.action_shape, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        self.gamma = cfg['agent']['gamma']
        self.tau = cfg['agent']['tau']

        state_flatten_shape = [np.prod(self.memory.flatten_state_shape)]
        # Actor Model
        self.actor = Actor(state_flatten_shape, self.action_shape, cfg['env']['action_range'],
                           self.tau, self.memory.batch_size, cfg['actor'])

        # Critic Model
        self.critic = Critic(state_flatten_shape, self.action_shape, self.tau, cfg['critic'])

        # Flag & Counter
        self.add_noise = True
        self.episode = 0
        self.max_episode_explore = 100

    def init_actor_critic(self):
        # Initialize target model
        self.critic.copy_local_in_target()
        self.actor.copy_local_in_target()

    def reset(self):
        self.memory.reset_past()
        self.noise = OUNoise(self.action_shape, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward,
                        next_state, done)
        if done:
            self.reset()


    def act(self, state):
        self.last_state = state

        window_states = self.memory.get_state_vector(state).reshape(1, -1)
        action = self.actor.predict(window_states)

        if self.add_noise and self.episode < self.max_episode_explore:
            p = self.episode / self.max_episode_explore
            action = np.clip(action*p + (1-p)*self.noise.sample(), a_max=1, a_min=-1)

        return action

    def learn(self):
        if self.memory.is_sufficient():
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
            critic_summaries = self.critic.fit(states, actions, Q_targets)

            # Train actor model
            action_gradients = self.critic.get_actions_grad(states, actions)[0]
            actor_summaries = self.actor.fit(states, action_gradients)

            # Soft-update target models
            self.critic.soft_update()
            self.actor.soft_update()

            summary_reward = summary('sample_rewards', rewards)

            return critic_summaries, actor_summaries, summary_reward

def summary(tag, values):

    summaries = [
        tf.Summary(value=[tf.Summary.Value(
            tag='memory/{}/min'.format(tag), simple_value=np.min(values)
        )]),
        tf.Summary(value=[tf.Summary.Value(
            tag='memory/{}/max'.format(tag), simple_value=np.max(values)
        )]),
        tf.Summary(value=[tf.Summary.Value(
            tag='memory/{}/mean'.format(tag), simple_value=np.mean(values)
        )]),
    ]

    return summaries