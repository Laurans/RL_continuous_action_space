from agents.agent import Agent
from Monitor import interact
import gym
import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

env = gym.make(cfg['env']['name'])

# Set a seed
env.seed(cfg['env']['seed'])
cfg['agent']['memory']['state_shape'] = list(env.observation_space.shape)
cfg['env']['action_shape'] = list(env.action_space.shape)
cfg['env']['action_range'] = [env.action_space.low, env.action_space.high]

agent = Agent(cfg)
avg_rewards, best_avg_reward = interact(env, agent,
                                        cfg['monitor']['num_episodes'],
                                        cfg['monitor']['window'])
