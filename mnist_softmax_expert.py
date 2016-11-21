from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def weight_variable(shape):
        """
        Returns set of weights, randomly initialized from a Gaussian
        distribution.
        This is for symmetry breaking, and to prevent 0 gradients.
        """
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

def bias_variable(shape):
        """
        Since we are using ReLU neurons, initialize with a small amount of
        positive bias to avoid "dead neurons".
        """
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

def conv2d(x, W):
        """
        Convolution with a stride of 1 and zero-padded, such that the output
        has the same dimensions as the input.
        """
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
        """
        Plain old max pooling over 2x2 blocks.
        """
        return tf.nn.max_pool(x,
                              ksize = [1, 2, 2, 1],
                              strides = [1, 2, 2, 1],
                              padding = 'SAME')

def conv_layer(shape, in_activations):
        """
        Creates a ConvNet layer with ReLU as activation function, and 2x2 max
        pooling.
        """
        W_conv = weight_variable(shape)
        b_conv = bias_variable([shape[-1]])

        h_conv = tf.nn.relu(conv2d(in_activations, W_conv) + b_conv)
        return max_pool_2x2(h_conv)

def mnist_softmax():
        """
        Models MNIST handwritten digit classification problem using a
        multi-layer convolutional neural network.

        The loss function is the cross-entropy, which internally computes
        softmax on the output activations.
        """
        mnist_data = input_data.read_data_sets("MNIST_data/", one_hot = True)

        x = tf.placeholder(tf.float32, shape = [None, 784])
        y_ = tf.placeholder(tf.float32, shape = [None, 10])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_pool1 = conv_layer([5, 5, 1, 32], x_image)
        h_pool2 = conv_layer([5, 5, 32, 64], h_pool1)

        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())

                for batch_index in range(20000):
                        batch_xs, batch_ys = mnist_data.train.next_batch(50)

                        if (batch_index % 100) == 0:
                                train_accuracy = accuracy.eval(feed_dict = {x: batch_xs,
                                                                            y_: batch_ys,
                                                                            keep_prob: 1.0})
                                print("step %d, training accuracy %g"%(batch_index, train_accuracy))

                        sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5})

                print(sess.run(accuracy, feed_dict = {x: mnist_data.test.images,
                                                      y_: mnist_data.test.labels,
                                                      keep_prob: 1.0}))

if __name__ == '__main__':
        mnist_softmax()
