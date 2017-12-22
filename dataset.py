from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
from six.moves import xrange

import tensorflow as tf

import arg_parsing

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

FLAGS = arg_parsing.parser.parse_args()

def download_and_extract():
    data_dir = FLAGS.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('\nSuccessfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(data_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(data_dir)


def _read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(
                  tf.strided_slice(record_bytes, [label_bytes],
                                   [label_bytes + image_bytes]),
                                   [result.depth, result.height, result.width])

    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def _generate_image_and_label_batch(image, label, min_q_eg, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                              [image, label],
                              batch_size=batch_size,
                              num_threads=num_preprocess_threads,
                              capacity=min_q_eg + 3 * batch_size,
                              min_after_dequeue=min_q_eg)
    else:
        images, label_batch = tf.train.batch(
                              [image, label],
                              batch_size=batch_size,
                              num_threads=num_preprocess_threads,
                              capacity=min_q_eg + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])


def process_inputs(mode):
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

    if mode == "training":
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(filenames)
    
    read_input = _read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    if mode == "training":
        reshaped_image = tf.random_crop(reshaped_image, [32, 32, 3])
        reshaped_image = tf.image.random_flip_left_right(reshaped_image)
        reshaped_image = tf.image.random_brightness(reshaped_image, max_delta=63)
        reshaped_image = tf.image.random_contrast(reshaped_image, lower=0.2, upper=1.8)
    else:
        reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 32, 32)

    float_image = tf.image.per_image_standardization(reshaped_image)
    float_image.set_shape([32, 32, 3])
    read_input.label.set_shape([1])

    min_queue_examples = int(0.4*arg_parsing.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

    shuffle = True if mode == "training" else False
    images, labels = _generate_image_and_label_batch(float_image, read_input.label,
                                                     min_queue_examples, FLAGS.batch_size,
                                                     shuffle=shuffle)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images, labels
