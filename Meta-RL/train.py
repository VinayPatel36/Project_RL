#Handling command line args

import argparse

parser = argparse.ArgumentParser()

#positional arguments
parser.add_argument('env_name',help='The Environment name from gym')
parser.add_argument('expected_reward',type=float , help='The reward at which Environment is considered solved')

#optional arguments
parser.add_argument('-ti','--train_iterations',type=int,help='Maximum training iterations')
parser.add_argument('-mel','--max_episode_length',type=int,help='Maximum steps in each episode')
parser.add_argument('-tbs','--trajectory_buffer_size',type=int,help='Trajectory Buffer Size')
parser.add_argument('-bs','--batch_size',type=int,help='Batch size')
parser.add_argument('-re','--render_every',type=int,help='Render every this many episodes')

args = parser.parse_args()


print(args)
ENV_NAME = args.env_name
EXPECTED_REWARD = args.expected_reward

#ENV_NAME = sys.argv[1]
TRAIN_ITERATIONS = 5000
MAX_EPISODE_LENGTH = 1000
TRAJECTORY_BUFFER_SIZE = 32
BATCH_SIZE = 16
RENDER_EVERY = 1
#EXPECTED_REWARD = int(sys.argv[2])

if args.train_iterations != None :
    TRAIN_ITERATIONS = args.train_iterations
if args.max_episode_length !=None :
    MAX_EPISODE_LENGTH = args.max_episode_length
if args.trajectory_buffer_size !=None :
    TRAJECTORY_BUFFER_SIZE = args.trajectory_buffer_size
if args.batch_size !=None :
    BATCH_SIZE = args.batch_size
if args.render_every !=None :
    RENDER_EVERY = args.render_every


#print('TRAIN_ITERATIONS',TRAIN_ITERATIONS,' MAX_EPISODE_LENGTH ',MAX_EPISODE_LENGTH)


import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
import tensorflow.keras as K
from Agent import *
import sys



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
