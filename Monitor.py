from collections import deque
import sys
import math
import numpy as np
import tensorflow as tf


def interact(env, agent, saver, sess, writer, num_episodes=20000, window=100, num_init_episodes=100, render=False):
    """ Monitor agent's performance.

    Params
    ======
    - env: instance of OpenAI Gym's environment
    - agent: instance of class Agent (see agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards

    # Random episodes
    for i_episode in range(1, num_init_episodes+1):
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)


            if done:
                next_state = np.zeros(state.shape)
                agent.memory.add(state, action, reward, next_state, done)
                break

            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
        print("\rEpisode {}/{} || RANDOM {}/{}".format(i_episode, num_init_episodes, len(agent.memory), agent.memory.max_size), end="\t")
        sys.stdout.flush()

    samp_rewards = deque(maxlen=window)
    step = 0
    reward_log = tf.Summary()
    loss_log = tf.Summary()
    # for each episode
    for i_episode in range(1, num_episodes + 1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        samp_loss = []
        while True:
            step += 1
            if render:
                env.render()
            # agent selects an action
            action = agent.act(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            loss = agent.step(action, reward, next_state, done)
            if loss is not None:
                samp_loss.append(loss)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)

                loss_log.value.add(tag='mean_episode_loss', simple_value=np.mean(samp_loss))
                reward_log.value.add(tag='episode_rewards', simple_value=samp_reward)
                writer.add_summary(loss_log, i_episode)
                writer.add_summary(reward_log, i_episode)

                break
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {} , last samp reward {}".format(i_episode, num_episodes, best_avg_reward, samp_reward), end="\t")
        sys.stdout.flush()

        if i_episode % 300 == 0:
            saver.save(sess, "checkpoints/{}_{}.ckpt".format('exp1', i_episode))
        if i_episode == num_episodes:
            saver.save(sess, "checkpoints/{}_{}.ckpt".format('exp1', 'final'))
            writer.close()
