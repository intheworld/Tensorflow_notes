import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]

A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
A_LR = 0.0001
C_LR = 0.0002
S_DIM, A_DIM = 3, 1


class PPO(object):

    def __init__(self, gamma=0.9, tau=0.1):
        self.sess = tf.Session()
        self.gamma = gamma
        self.tau = tau
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.tfs_ = tf.placeholder(tf.float32, [None, S_DIM], 'state_')
        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.ret_ph = tf.placeholder(tf.float32, [None, 1], name='v')

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        act = tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), -2, 2)

        # critic
        self.critic_net = self._build_critic_net(scope='critic')

        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

        with tf.variable_scope('critic_loss'):
            self.critic_loss_op = tf.reduce_mean(tf.squared_difference(self.ret_ph, self.critic_net))

        with tf.variable_scope('critic_train'):
            self.critic_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.critic_loss_op, var_list=self.ce_params)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r, bv):
        self.sess.run(self.update_oldpi_op)
        v = self.sess.run(self.critic_net, {self.tfs: s})
        adv = bv - v
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)   # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.critic_train_op, {self.tfs: s, self.ret_ph: bv}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_critic_net(self, scope):
        h1_units = 100
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(self.tfs, h1_units, tf.nn.relu)
            critic_net = tf.layers.dense(l1, 1)
        return critic_net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.critic_net, {self.tfs: s})[0, 0]

