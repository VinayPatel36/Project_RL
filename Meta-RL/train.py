import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
import tensorflow.keras as K
from Agent import *
import sys


ENV_NAME = sys.argv[1]
TRAIN_ITERATIONS = 5000
MAX_EPISODE_LENGTH = 1000
TRAJECTORY_BUFFER_SIZE = 32
BATCH_SIZE = 16
RENDER_EVERY = 1
EXPECTED_REWARD = int(sys.argv[2])

env = gym.make(ENV_NAME)
agent = Agent(env.action_space.n,env.observation_space.shape,BATCH_SIZE)
samples_filled = 0

scores_window = deque(maxlen=100)
scores = []
max_reward = -500
for cnt_episode in range(TRAIN_ITERATIONS):
    s = env.reset()
    r_sum = 0
    for cnt_step in range(MAX_EPISODE_LENGTH):
        if cnt_episode % RENDER_EVERY == 0 :
            env.render()
        a = agent.choose_action(s)
        s_, r, done, _ = env.step(a)
        r_sum += r
        agent.store_transition(s, a, s_, r, done)
        samples_filled += 1
        if samples_filled % TRAJECTORY_BUFFER_SIZE == 0 and samples_filled != 0:
            for _ in range(TRAJECTORY_BUFFER_SIZE // BATCH_SIZE):
                agent.train_network()
            agent.memory.clear()
            samples_filled = 0
        s = s_
        if done:
            break
    scores_window.append(r_sum)
    scores.append(r_sum)
    if np.mean(scores_window)>=EXPECTED_REWARD:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(cnt_episode-100, np.mean(scores_window)))
        agent.actor_network.save_weights(str(r_sum)+"acrobot_actor.h5")
        break
    max_reward = max(max_reward, r_sum)
    print('Episodes:', cnt_episode, 'Episodic_Reweard:', r_sum, 'Max_Reward_Achieved:', max_reward)
