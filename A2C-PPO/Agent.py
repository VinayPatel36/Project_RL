import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
import tensorflow.keras as K
from Memory import *

class Agent:
    def __init__(self,action_n, state_dim, training_batch_size):
        self.action_n = action_n
        self.state_dim = state_dim
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.9
        self.critic_network = self._build_critic_network()
        self.actor_network = self._build_actor_network()
        self.actor_old_network = self._build_actor_network()
        self.actor_old_network.set_weights(self.actor_network.get_weights())
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, self.action_n))
        self.memory = Memory()



    def _build_actor_network(self):
        state = K.layers.Input(shape=self.state_dim,name='state_input')
        advantage = K.layers.Input(shape=(1,),name='advantage_input')
        old_prediction = K.layers.Input(shape=(self.action_n,),name='old_prediction_input')
        rnn_in = tf.expand_dims(state, [0])
        lstm = K.layers.LSTM(24,activation='relu')(rnn_in)
        dense = K.layers.Dense(32,activation='relu',name='dense1')(lstm)
        dense = K.layers.Dense(32,activation='relu',name='dense2')(dense)
        policy = K.layers.Dense(self.action_n, activation="softmax", name="actor_output_layer")(dense)
        actor_network = K.Model(inputs = [state,advantage,old_prediction], outputs = policy)
        actor_network.compile(
            optimizer='Adam',
            loss = self.ppo_loss(advantage=advantage,old_prediction=old_prediction)
            )
        actor_network.summary()
        time.sleep(1.0)
        return actor_network


    def _build_critic_network(self):
        state = K.layers.Input(shape=self.state_dim,name='state_input')
        dense = K.layers.Dense(32,activation='relu',name='dense1')(state)
        dense = K.layers.Dense(32,activation='relu',name='dense2')(dense)
        V = K.layers.Dense(1, name="actor_output_layer")(dense)
        critic_network = K.Model(inputs=state, outputs=V)
        critic_network.compile(optimizer='Adam',loss = 'mean_squared_error')
        critic_network.summary()
        time.sleep(1.0)
        return critic_network


    def ppo_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            ratio = prob / (old_prob + 1e-10)
            clip_ratio = K.backend.clip(ratio, min_value=1 - self.CLIPPING_LOSS_RATIO, max_value=1 + self.CLIPPING_LOSS_RATIO)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            entropy_loss = (prob * K.backend.log(prob + 1e-10))
            ppo_loss = -K.backend.mean(K.backend.minimum(surrogate1,surrogate2) + self.ENTROPY_LOSS_RATIO * entropy_loss)
            return ppo_loss
        return loss


    def make_gae(self):
        gae = 0
        mask = 0
        for i in reversed(range(self.memory.cnt_samples)):
            mask = 0 if self.memory.batch_done[i] else 1
            v = self.get_v(self.memory.batch_s[i])
            delta = self.memory.batch_r[i] + self.GAMMA * self.get_v(self.memory.batch_s_[i]) * mask - v
            gae = delta + self.GAMMA *  self.GAE_LAMBDA * mask * gae
            self.memory.batch_gae_r.append(gae+v)
        self.memory.batch_gae_r.reverse()
        self.memory.GAE_CALCULATED_Q = True


    def update_tartget_network(self):
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.actor_network.get_weights())
        actor_tartget_weights = np.array(self.actor_old_network.get_weights())
        new_weights = alpha*actor_weights + (1-alpha)*actor_tartget_weights
        self.actor_old_network.set_weights(new_weights)


    def choose_action(self,state):
        assert isinstance(state,np.ndarray)
        state = np.reshape(state,[-1,self.state_dim[0]])
        prob = self.actor_network.predict_on_batch([state,self.dummy_advantage, self.dummy_old_prediciton]).flatten()
        action = np.random.choice(self.action_n,p=prob)
        return action


    def train_network(self):
        if not self.memory.GAE_CALCULATED_Q:
            self.make_gae()
        states,actions,rewards,gae_r,next_states,dones = self.memory.get_batch(self.TRAINING_BATCH_SIZE)

        batch_s = np.vstack(states)
        batch_a = np.vstack(actions)
        batch_gae_r = np.vstack(gae_r)
        batch_v = self.get_v(batch_s)
        batch_advantage = batch_gae_r - batch_v
        batch_advantage = K.utils.normalize(batch_advantage) #
        batch_old_prediction = self.get_old_prediction(batch_s)
        batch_a_final = np.zeros(shape=(len(batch_a), self.action_n))
        batch_a_final[:, batch_a.flatten()] = 1

        self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)
        self.critic_network.fit(x=batch_s, y=batch_gae_r, epochs=1, verbose=0)
        self.update_tartget_network()


    def store_transition(self, s, a, s_, r, done):
        self.memory.store(s, a, s_, r, done)


    def get_v(self,state):
        s = np.reshape(state,(-1, self.state_dim[0]))
        v = self.critic_network.predict_on_batch(s)
        return v


    def get_old_prediction(self, state):
        state = np.reshape(state, (-1, self.state_dim[0]))
        return self.actor_old_network.predict_on_batch([state,self.dummy_advantage, self.dummy_old_prediciton])
