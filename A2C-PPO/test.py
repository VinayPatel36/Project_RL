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
WEIGHT_FILE = sys.argv[2]

env = gym.make(ENV_NAME)
agent = Agent(env.action_space.n,env.observation_space.shape,BATCH_SIZE)
samples_filled = 0

agent.actor_network.load_weights(WEIGHT_FILE)

for i in range(15):
    episode_reward = 0
    state = env.reset()
    while True:
        action = agent.choose_action(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break
        episode_reward += reward
    print('Episodes:', i, 'Episodic_Reweard:', episode_reward)
env.close()
