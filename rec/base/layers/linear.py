import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


# from keras.premade.linear import LinearModel


class Linear(keras.layers.Layer):
    """
    from keras.premade.linear import LinearModel
    """

    def __init__(self, output_dim, use_bias=True):
        super(Linear, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias

    def build(self, input_shape):
        # names = sorted(list(input_shape.keys()))
        input_dim = input_shape[-1]
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(self.output_dim,),
                                        )
        self.kernel = self.add_weight(name='linear_kernel',
                                      shape=(input_dim, self.output_dim),
                                      )
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        linear_out = tf.tensordot(inputs, self.kernel, axes=[[-1], [0]])
        if self.use_bias:
            linear_out += self.bias
        return linear_out

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'use_bias': self.bias}
        base_config = super(Linear, self).get_config()
        for k, v in base_config.items():
            config[k] = v
        return config


def main():
    x = tf.constant([[1, 2, 3, 4, 5, 6]], dtype=tf.float32)
    linear = keras.layers.Dense(units=2)
    #
    x = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
    linear = Linear(output_dim=2)
    #

    #
    y_pred = linear(x)
    print(y_pred)


if __name__ == '__main__':
    main()
