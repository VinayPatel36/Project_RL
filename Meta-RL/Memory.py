import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
import tensorflow.keras as K


class Memory:
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_gae_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.GAE_CALCULATED_Q = False


    def get_batch(self,batch_size):
        for _ in range(batch_size):
            s,a,r,gae_r,s_,d = [],[],[],[],[],[]
            pos = np.random.randint(len(self.batch_s))
            s.append(self.batch_s[pos])
            a.append(self.batch_a[pos])
            r.append(self.batch_r[pos])
            gae_r.append(self.batch_gae_r[pos])
            s_.append(self.batch_s_[pos])
            d.append(self.batch_done[pos])
        return s,a,r,gae_r,s_,d


    def store(self, s, a, s_, r, done):
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)


    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.GAE_CALCULATED_Q = False


    @property
    def cnt_samples(self):
        return len(self.batch_s)
