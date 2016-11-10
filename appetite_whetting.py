"""
Generates some data in two dimensions, then fits it to a line.
"""
import tensorflow as tf
import numpy as np

def whet_appetite():
    """
    Whets appetite.
    """
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))

if __name__ == '__main__':
    whet_appetite()
