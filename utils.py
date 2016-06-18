import tensorflow as tf
import numpy as np

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum

        self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
        self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            self.beta = tf.get_variable(self.name+"beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
            self.gamma = tf.get_variable(self.name+"gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema_apply_op = self.ema.apply([batch_mean, batch_var])
            self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

            with tf.control_dependencies([ema_apply_op]):
                mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed
