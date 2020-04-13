import tensorflow as tf
import scipy.signal
import numpy as np
import gym
from ac_network import AC_Network
from collections import deque

scores = []
scores_window = deque(maxlen=100)
MINI_BATCH = 30
REWARD_FACTOR = 0.001

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def weighted_pick(weights,n_picks):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t,np.random.rand(n_picks)*s)

def discounting(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def norm(x, upper, lower=0.):
    return (x-lower)/max((upper-lower), 1e-12)

class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes, env_name, seed, test):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.is_test = test
        self.a_size = a_size

        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = gym.make(env_name)
        self.env.seed(seed)

    def get_env(self):
        return self.env

    def train(self, rollout, sess, gamma, r):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        rewards_list = np.asarray(rewards.tolist()+[r])*REWARD_FACTOR
        discounted_rewards = discounting(rewards_list, gamma)[:-1]

        values_list = np.asarray(values.tolist()+[r])*REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = discounting(advantages, gamma)


        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(states),
                     self.local_AC.actions: np.vstack(actions),
                     self.local_AC.advantages: discounted_advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_mini_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0

                terminal = False
                s = self.env.reset()

                rnn_state = self.local_AC.state_init

                while not terminal:
                    episode_states.append(s)
                    if self.is_test:
                        self.env.render()

                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                         feed_dict={self.local_AC.inputs: [s],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})

                    a0 = weighted_pick(a_dist[0], 1)
                    if self.is_test:
                        a0 = np.argmax(a_dist[0])
                    a = np.zeros(self.a_size)
                    a[a0] = 1

                    s2, r, terminal, info = self.env.step(np.argmax(a))

                    episode_reward += r

                    episode_buffer.append([s, a, r, s2, terminal, v[0, 0]])
                    episode_mini_buffer.append([s, a, r, s2, terminal, v[0, 0]])

                    episode_values.append(v[0, 0])

                    if len(episode_mini_buffer) == MINI_BATCH and not self.is_test:
                        v1 = sess.run([self.local_AC.value],
                                      feed_dict={self.local_AC.inputs: [s],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_mini_buffer, sess, gamma, v1[0][0])
                        episode_mini_buffer = []

                    s = s2
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if episode_count % 10 == 0 and not episode_count % 100 == 0 and not self.is_test:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    if episode_count % 100 == 0 and not self.is_test:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')

                    print("| Reward: " + str(episode_reward), " | Episode", episode_count)
                    scores_window.append(episode_reward)
                    if np.mean(scores_window)>=-110.0:
                        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_count-100, np.mean(scores_window)))
                        coord.request_stop()
                    sess.run(self.increment)

                episode_count += 1
