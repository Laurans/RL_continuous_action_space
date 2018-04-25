from collections import deque
import sys
import math
import numpy as np
import tensorflow as tf


def training(env, agent, saver, sess, writer, num_episodes=20000, window=100, num_init_episodes=100, render=False,
             exp_name='exp'):
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
    # initialize best average reward
    best_avg_reward = -math.inf
    best_reward = -math.inf

    episode_rewards_history = deque(maxlen=100)
    iterations = 0

    # for each episode
    for i_episode in range(1, num_episodes+1):
        episode_steps = 0
        episode_score = 0
        state = env.reset()
        agent.reset()
        agent.add_noise = True

        while True:
            # agent selects an action
            action = agent.act(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            agent.step(action, reward, next_state, done)

            # LEARNING ----------------------------------------------------------------
            learning_summaries = agent.learn()
            if learning_summaries is not None:
                for s in learning_summaries:
                    if type(s) == list:
                        for e in s:
                            writer.add_summary(e, iterations)
                    else:
                        writer.add_summary(s, iterations)
            # UPDATE METRICS ----------------
            episode_score += reward
            episode_steps += 1
            iterations += 1
            # update the state (s <- s') to next time step
            state = next_state
            # DONE --------------
            if done:
                agent.episode += 1
                episode_rewards_history.append(episode_score)
                best_reward = max(best_reward, episode_score)
                break

        if i_episode > 100:
            best_avg_reward = max(best_avg_reward, np.mean(episode_rewards_history))

        print("Iter {} | Episode {}/{} | Score {} | Step {} \t Best avg reward {} | Best reward {}".format(
            iterations, i_episode, num_episodes, episode_score, episode_steps, best_avg_reward, best_reward))

        train_score = tf.Summary()
        train_score.value.add(tag='train/score', simple_value=episode_score)
        writer.add_summary(train_score, i_episode)

        train_best_avg = tf.Summary()
        train_best_avg.value.add(tag='train/best_avg_score', simple_value=best_avg_reward)
        writer.add_summary(train_best_avg, i_episode)

        # TESTING ----------------------------------------------------------------
        agent.add_noise = False
        state = env.reset()
        agent.reset()
        episode_score = 0
        episode_steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(action, reward, next_state, done)
            episode_score += reward
            episode_steps += 1
            state = next_state
            if done:
                print("Testing at {} | Reward {} | Step {}".format(i_episode, episode_score, episode_steps))

                eval_score = tf.Summary()
                eval_score.value.add(tag='eval/score', simple_value=episode_score)
                writer.add_summary(eval_score, i_episode)

                eval_episode_steps = tf.Summary()
                eval_episode_steps.value.add(tag='eval/episode_steps', simple_value=episode_steps)
                writer.add_summary(eval_episode_steps, i_episode)

                break
        print("Memory {}/{} | Mean {} | Min {} | Max {}".format(
            len(agent.memory), len(agent.memory.memory), agent.memory.mean(), agent.memory.min(), agent.memory.max()))
        print()
        if i_episode % 50 == 0:
            saver.save(sess, "checkpoints/{}/{}.ckpt".format(exp_name, i_episode))
    saver.save(sess, "checkpoints/{}/{}.ckpt".format(exp_name, 'final'))
    writer.close()

