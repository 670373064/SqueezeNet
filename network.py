from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

import tensorflow as tf

import arg_parsing

def _conv_module(inputs, num_outputs, kernel_size, stride, scope, activation_fn=tf.nn.relu):
    with tf.variable_scope(scope, 'Conv', [inputs]) as sc:
        kSize = kernel_size+[inputs.get_shape().as_list()[3], num_outputs]
        kernel = tf.Variable(tf.random_normal(kSize))
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME')
        biases = tf.Variable(tf.random_normal([num_outputs]))
        conv1 = tf.nn.bias_add(conv, biases)
        if activation_fn is not None:
            conv1 = activation_fn(conv1, name=sc.name)

    return conv1

def _fire_module(inputs, squeeze_depth, expand_depth, scope):
    with tf.variable_scope(scope, 'Fire', [inputs]) as sc:
        net = _conv_module(inputs, squeeze_depth, [1, 1], 1, scope='squeeze')
        with tf.variable_scope('expand'):
            e1x1 = _conv_module(net, expand_depth, [1, 1], 1, scope='1x1')
            e3x3 = _conv_module(net, expand_depth, [3, 3], 1, scope='3x3')
            net = tf.concat([e1x1, e3x3], 3)
    net = tf.clip_by_norm(net, 100)
    return net

def inference(images):
    net = _conv_module(images, 96, [2, 2], 1, 'conv1')
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    net = _fire_module(net, 16, 64, 'fire2')
    net = _fire_module(net, 16, 64, 'fire3')
    net = _fire_module(net, 32, 128, 'fire4')
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    net = _fire_module(net, 32, 128, 'fire5')
    net = _fire_module(net, 48, 192, 'fire6')
    net = _fire_module(net, 48, 192, 'fire7')
    net = _fire_module(net, 64, 256, 'fire8')
    net = tf.nn.max_pool(net, [1, 4, 4, 1], [1, 2, 2, 1], 'SAME')
    net = _fire_module(net, 64, 256, 'fire9')
    net = tf.nn.avg_pool(net, [1, 4, 4, 1], [1, 4, 4, 1], 'SAME')
    net = _conv_module(net, 10, [1, 1], 1, 'conv10', None)
    logits = tf.squeeze(net, [1, 2], name='logits')

    return logits
