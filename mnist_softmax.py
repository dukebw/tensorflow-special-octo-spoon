from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def mnist_softmax():
        """
        Models MNIST handwritten digit classification problem using matrix
        multiplication y = W*x + b.

        The loss function is the cross-entropy, which internally computes
        softmax on the output activations.
        """
        mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b

        y_ = tf.placeholder(tf.float32, [None, 10])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.InteractiveSession()

        tf.initialize_all_variables().run()
        for _ in range(1000):
                batch_xs, batch_ys = mnist_data.train.next_batch(100)
                sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict = {x: mnist_data.test.images,
                                              y_: mnist_data.test.labels}))

if __name__ == '__main__':
        mnist_softmax()
