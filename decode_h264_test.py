import tensorflow as tf
from PIL import Image
decode_h264 = tf.load_op_library('decode_h264.so')

def test_decode_h264():
    """
    """
    with tf.Session() as sess:
        reader = tf.WholeFileReader()
        filename_queue = tf.train.string_input_producer(['temp0.mp4'])
        _, test_mp4 = reader.read(filename_queue)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        decoded_tensor = decode_h264.decode_h264(test_mp4, num_frames=30*10)
        decoded_mp4 = sess.run(decoded_tensor)
        Image.fromarray(decoded_mp4[0]).show()

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    test_decode_h264()
