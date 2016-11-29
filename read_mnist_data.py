import gzip
import numpy
import os
import tensorflow as tf

class DataSet:
        def __init__(self, images, labels, num_examples):
                self._images = images
                self._labels = labels
                self._num_examples = num_examples

        @property
        def images(self):
                return self._images

        @property
        def labels(self):
                return self._labels

        @property
        def num_examples(self):
                return self._num_examples

def read32_big_endian(byte_stream):
        big_endian_uint32 = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(byte_stream.read(4), dtype = big_endian_uint32)[0]

def int64_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def bytes_feature(value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def read_mnist_data(image_filename, labels_filename):
        """
        @desc Reads MNIST data (images and labels) and returns them as numpy
        arrays in a DataSet type.

        @param image_filename Filename of gzipped images.
        @param labels_filename Filename of gzipped labels.

        @return A DataSet type containing the images and labels from the
        dataset, as numpy arrays.
        """
        MNIST_DIR = './MNIST_data/'
        image_filepath = os.path.join(MNIST_DIR, image_filename)
        with gzip.open(image_filepath, 'rb') as images_stream:
                magic = read32_big_endian(images_stream)
                assert magic == 2051
                num_images = read32_big_endian(images_stream)
                num_rows = read32_big_endian(images_stream)
                num_columns = read32_big_endian(images_stream)
                images = numpy.frombuffer(images_stream.read(), dtype = numpy.uint8)
                images = numpy.reshape(images, [num_images, num_rows, num_columns])

        labels_filepath = os.path.join(MNIST_DIR, labels_filename)
        with gzip.open(labels_filepath, 'rb') as labels_stream:
                magic = read32_big_endian(labels_stream)
                assert magic == 2049
                num_items = read32_big_endian(labels_stream)
                labels = numpy.frombuffer(labels_stream.read(), dtype = numpy.uint8)

        assert num_images == num_items

        return DataSet(images, labels, num_images)

def write_tfrecords(dataset, tf_records_filepath, filename):
        filename = os.path.join(tf_records_filepath, filename + '.tfrecords')
        print('Writing', filename)

        with tf.python_io.TFRecordWriter(filename) as writer:
                for example_index in range(dataset.num_examples):
                        image_raw = dataset.images[example_index].tostring()
                        feature = {
                                'height': int64_feature(dataset.images.shape[1]),
                                'width': int64_feature(dataset.images.shape[2]),
                                'label': int64_feature(int(dataset.labels[example_index])),
                                'image_raw': bytes_feature(image_raw)
                        }
                        next_features = tf.train.Features(feature = feature)
                        example = tf.train.Example(features = next_features)
                        writer.write(example.SerializeToString())

def convert_data_to_tf_record(tf_records_filepath):
        # TODO(brendan): Cross-validation data
        train_data = read_mnist_data('train-images-idx3-ubyte.gz',
                                     'train-labels-idx1-ubyte.gz')

        write_tfrecords(train_data, tf_records_filepath, 'train')

        test_data = read_mnist_data('t10k-images-idx3-ubyte.gz',
                                    't10k-labels-idx1-ubyte.gz')

        write_tfrecords(test_data, tf_records_filepath, 'test')

def main(argv):
        assert len(argv) == 1
        convert_data_to_tf_record(argv[0])

if __name__ == "__main__":
        tf.app.run()
