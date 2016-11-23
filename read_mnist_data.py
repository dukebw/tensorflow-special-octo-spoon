import gzip
import numpy

BIG_ENDIAN_UINT32 = numpy.dtype(numpy.uint32).newbyteorder('>')

class DataSet:
        def __init__(self, images, labels):
                self._images = images
                self._labels = labels

        @property
        def images(self):
                return self._images

        @property
        def labels(self):
                return self._labels

def read32_big_endian(byte_stream):
        return numpy.frombuffer(byte_stream.read(4), dtype = BIG_ENDIAN_UINT32)

def read_mnist_data():
        with gzip.open('./MNIST_data/train-images-idx3-ubyte.gz', 'rb') as images_stream:
                magic = read32_big_endian(images_stream)
                assert magic == 2051
                num_images = read32_big_endian(images_stream)
                num_rows = read32_big_endian(images_stream)
                num_columns = read32_big_endian(images_stream)
                images = numpy.frombuffer(images_stream.read(), dtype = numpy.uint8)
                images = numpy.reshape(images, [num_images, num_rows, num_columns])

if __name__ == "__main__":
        read_mnist_data()
