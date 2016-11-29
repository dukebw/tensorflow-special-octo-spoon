import tensorflow as tf

def read_decode_single(filename_queue):
        """
        @desc Reads and decodes a single example protocol buffer from
        filename_queue as a tensor, which can be used for input to
        tf.train.shuffle_batch().

        @param [in] filename_queue Queue from which the example protocol buffer
        should be dequeued.

        @return (image, label) example tuple of tensors from the MNIST dataset.
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
        }
        features = tf.parse_single_example(serialized_example, features)

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([28*28])

        image = (1./255)*tf.cast(image, tf.float32) - 0.5

        label = tf.cast(features['label'], tf.int32)

        return image, label

def main(argv):
        with tf.name_scope('input'):
                filename_queue = tf.train.string_input_producer(["train.tfrecords"])

                image, label = read_decode_single(filename_queue)

                BATCH_SIZE = 100
                SAMPLE_BUFFER_SIZE = 10000
                images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                               batch_size = BATCH_SIZE,
                                                               capacity = SAMPLE_BUFFER_SIZE + 3*BATCH_SIZE,
                                                               min_after_dequeue = SAMPLE_BUFFER_SIZE,
                                                               num_threads = 2)

if __name__ == "__main__":
        tf.app.run()
