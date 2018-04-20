# REMOVE VERBOSE
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import warnings
warnings.filterwarnings('ignore')

from agents.agent import Agent
from Monitor import interact
import gym
import yaml
import tensorflow as tf





with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

env = gym.make(cfg['env']['name'])

# Set a seed
env.seed(cfg['env']['seed'])
tf.set_random_seed(cfg['env']['seed'])


cfg['agent']['memory']['state_shape'] = list(env.observation_space.shape)
cfg['env']['action_shape'] = list(env.action_space.shape)
cfg['env']['action_range'] = [env.action_space.low, env.action_space.high]

if cfg['monitor']['render']:
    print('With visualization')
else:
    print('Without visualization')

with tf.Session() as sess:
    agent = Agent(cfg)
    writer = tf.summary.FileWriter("./logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    agent.init_actor_critic()

    saver = tf.train.Saver()
    interact(env, agent, saver, sess, writer,
             cfg['monitor']['num_episodes'],
             cfg['monitor']['window'],
             cfg['monitor']['num_init_episodes'],
             cfg['monitor']['render'])
