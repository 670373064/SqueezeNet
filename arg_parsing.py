from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 5000
LEARNING_RATE_DECAY_FACTOR = 0.75
INITIAL_LEARNING_RATE = 0.01

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='training',
                    help='Either `training` or `testing`.')
parser.add_argument('--data_dir', type=str, default='/home/ubuntu/squeeze/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')
parser.add_argument('--model_dir', type=str, default='/home/ubuntu/squeeze/cifar10_model',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')
parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')
parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of batches to run.')
parser.add_argument('--log_frequency', type=int, default=100,
                    help='How often to log results to the console.')
parser.add_argument('--num_examples', type=int, default=10000,
                    help='Number of examples to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')
