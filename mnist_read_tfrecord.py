import tensorflow as tf

def main(argv):
        assert len(argv) == 0

        with tf.name_scope('input'):
                filename_queue = tf.train.string_input_producer("train.tfrecords")

                # TODO(brendan): read and decode

if __name__ == "__main__":
        tf.app.run()
