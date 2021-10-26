from tensorflow_probability import distributions as tfpd
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
from collections import defaultdict

class DoubleheadModel(tf.keras.Sequential):
    distribution = tfpd.Normal
    def __init__(self, input_shape, hidden):
        self.input_shape_ = input_shape
        self.hidden = hidden

        self.max_logstd = tf.Variable(
            tf.fill([1, 1], np.log(0.2).astype(np.float32)), trainable=True
        )
        self.min_logstd = tf.Variable(
            tf.fill([1, 1], np.log(0.1).astype(np.float32)), trainable=True
        )

        layers = [
            tfkl.Flatten(input_shape=input_shape),
            tfkl.Dense(hidden, activation="softplus"),
            tfkl.Dense(hidden, activation="softplus"),
            tfkl.Dense(2),
        ]

        super(DoubleheadModel, self).__init__(layers)

    def get_params(self, inputs, **kwargs):
        prediction = super(DoubleheadModel, self).__call__(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale": tf.math.exp(logstd)}

    def get_distribution(self, inputs, **kwargs):
        return self.distribution(**self.get_params(inputs, **kwargs))

class SingleheadModel(tf.keras.Sequential):
    distribution = tfpd.Normal
    def __init__(self, input_shape, hidden):
        self.input_shape_ = input_shape
        self.hidden = hidden

        layers = [
            tfkl.Flatten(input_shape=input_shape),
            tfkl.Dense(hidden, activation=tfkl.LeakyReLU()),
            tfkl.Dense(hidden, activation=tfkl.LeakyReLU()),
            tfkl.Dense(1),
        ]

        super(SingleheadModel, self).__init__(layers)

    def get_params(self, inputs, **kwargs):
        prediction = super(SingleheadModel, self).__call__(inputs, **kwargs)
        return {"loc": prediction, "scale": 1.0}

    def get_distribution(self, inputs, **kwargs):
        return self.distribution(**self.get_params(inputs, **kwargs))


class NemoModel(tf.keras.Model):
    distribution = tfpd.Normal
    def __init__(self, input_shape, hidden):
        super(NemoModel, self).__init__()
        self.input_shape_ = input_shape
        self.hidden = hidden

        self.max_logstd = tf.Variable(
            tf.fill([1, 1], np.log(0.2).astype(np.float32)), trainable=True
        )
        self.min_logstd = tf.Variable(
            tf.fill([1, 1], np.log(0.1).astype(np.float32)), trainable=True
        )

        self.flatten = tfkl.Flatten(input_shape=input_shape)
        self.dense0 = tfkl.Dense(hidden, activation="softplus")
        self.dense1 = tfkl.Dense(hidden, activation="softplus")
        self.dense2 = tfkl.Dense(2)

    def call(self, inputs, **kwargs):
        out = self.flatten(inputs, **kwargs)
        out = self.dense0(out, **kwargs)
        out = out + self.dense1(out, **kwargs)
        out = self.dense2(out, **kwargs)
        return out

    def get_params(self, inputs, **kwargs):
        prediction = self.call(inputs, **kwargs)
        mean, logstd = tf.split(prediction, 2, axis=-1)
        logstd = self.max_logstd - tf.nn.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + tf.nn.softplus(logstd - self.min_logstd)
        return {"loc": mean, "scale": tf.math.exp(logstd)}

    def get_distribution(self, inputs, **kwargs):
        return self.distribution(**self.get_params(inputs, **kwargs))