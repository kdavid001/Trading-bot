import tensorflow as tf
from tensorflow.keras.layers import Layer


class Attention(Layer):
    """
    Custom attention layer for temporal data
    """

    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output
        return tf.reduce_sum(output, axis=1)

    def get_config(self):
        return {'return_sequences': self.return_sequences}