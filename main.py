from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

import arg_parsing
import dataset
import train
import test

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(argv=None):

    FLAGS = arg_parsing.parser.parse_args()
    dataset.download_and_extract()

    if (FLAGS.mode == 'training'):
        if tf.gfile.Exists(FLAGS.model_dir):
            tf.gfile.DeleteRecursively(FLAGS.model_dir)
        tf.gfile.MakeDirs(FLAGS.model_dir)
        train.train()

    elif (FLAGS.mode == 'testing'):
        test.test()
    else:
        raise ValueError("set --mode as 'training' or 'testing'")

if __name__ == '__main__':
    tf.app.run()
