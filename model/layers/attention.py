import tensorflow as tf
from tensorflow.keras import layers
from layers.activation_func import Dice, dice


class attention(tf.keras.layers.Layer):
    def __init__(self, keys_dim):
        super(attention, self).__init__()
        self.keys_dim = keys_dim
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(36, activation='sigmoid'))
        self.fc.add(dice(36))
        self.fc.add(layers.Dense(1, activation=None))

    def call(self, queries, keys, keys_length):
        print('starting attention')
        # Attention
        queries = tf.tile(tf.expand_dims(queries, 1), [1, tf.shape(keys)[1], 1])
        print('queris -> shape', queries.shape, '\n')
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        outputs = tf.transpose(self.fc(din_all), [0, 2, 1])
        key_masks = tf.sequence_mask(keys_length, max(keys_length), dtype=tf.bool)
        key_masks = tf.expand_dims(key_masks, 1)
        paddings = tf.ones_like(outputs) * (-2**32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)
        outputs = outputs / (self.keys_dim ** 0.5)
        outputs = tf.keras.activations.sigmoid(outputs)

        print('==============attention================')
        print('outputs -> ', outputs.shape, '\n', outputs, '\n')
        print('keys -> ', keys.shape, '\n', keys, '\n')
        print('starting  tf.matmul')
        # Sum pooling
        outputs = tf.squeeze(tf.matmul(outputs, keys))
        print("outputs: " + str(outputs.numpy().shape))
        return outputs


class sum_pooling(tf.keras.layers.Layer):
    def __init__(self, axis):
        super(sum_pooling, self).__init__()
        self.axis = axis

    def call(self, queries):
        return tf.reduce_sum(queries, axis=self.axis)