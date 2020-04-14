import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, LSTM, Reshape, Flatten, GRU,SimpleRNN
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
tf.set_random_seed(2212)

class Actor:
    def __init__(self, sess, action_dim, observation_dim):
        self.action_dim, self.observation_dim = action_dim, observation_dim
        K.set_session(sess)
        self.sess = sess
        self.state_input, self.output, self.model = self.create_model()
        self.advantages = tf.placeholder(tf.float32, shape=[None, action_dim])
        model_weights = self.model.trainable_weights
        log_prob = tf.math.log(self.output + 10e-10)
        neg_log_prob = tf.multiply(log_prob, -1)
        actor_gradients = tf.gradients(neg_log_prob, model_weights, self.advantages)
        grads = zip(actor_gradients, model_weights)
        self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        rnn_in = tf.expand_dims(state_input, [0])
        lstm = LSTM(128)(rnn_in)
        state_h1 = Dense(24, activation='relu')(lstm)

        state_h2 = Dense(24, activation='relu')(state_h1)
        output = Dense(self.action_dim, activation='softmax')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        return state_input, output, model

    def train(self, X, y):
        self.sess.run(self.optimize, feed_dict={self.state_input:X, self.advantages:y})
