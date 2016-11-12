from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def mnist_softmax():
        mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == '__main__':
        mnist_softmax()
