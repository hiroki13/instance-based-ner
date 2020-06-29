# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def highway_layer(inputs, use_bias=True, bias_init=0.0, keep_prob=1.0,
                  is_train=False, scope=None):
  with tf.variable_scope(scope or "highway_layer"):
    hidden = inputs.get_shape().as_list()[-1]
    with tf.variable_scope("trans"):
      trans = tf.layers.dropout(inputs, rate=1.0 - keep_prob,
                                training=is_train)
      trans = tf.layers.dense(trans, units=hidden, use_bias=use_bias,
                              bias_initializer=tf.constant_initializer(
                                bias_init), activation=None)
      trans = tf.nn.relu(trans)
    with tf.variable_scope("gate"):
      gate = tf.layers.dropout(inputs, rate=1.0 - keep_prob, training=is_train)
      gate = tf.layers.dense(gate, units=hidden, use_bias=use_bias,
                             bias_initializer=tf.constant_initializer(
                               bias_init), activation=None)
      gate = tf.nn.sigmoid(gate)
  outputs = gate * trans + (1 - gate) * inputs
  return outputs


def highway_network(inputs, highway_layers=2, use_bias=True, bias_init=0.0,
                    keep_prob=1.0, is_train=False, scope=None):
  with tf.variable_scope(scope or "highway_network"):
    prev = inputs
    cur = None
    for idx in range(highway_layers):
      cur = highway_layer(prev, use_bias, bias_init, keep_prob, is_train,
                          scope="highway_layer_{}".format(idx))
      prev = cur
    return cur


def conv1d(in_, filter_size, height, padding, is_train=True, drop_rate=0.0,
           scope=None):
  with tf.variable_scope(scope or "conv1d"):
    num_channels = in_.get_shape()[-1]
    filter_ = tf.get_variable("filter",
                              shape=[1, height, num_channels, filter_size],
                              dtype=tf.float32)
    bias = tf.get_variable("bias", shape=[filter_size], dtype=tf.float32)
    strides = [1, 1, 1, 1]
    in_ = tf.layers.dropout(in_, rate=drop_rate, training=is_train)
    # [batch, max_len_sent, max_len_word / filter_stride, char output size]
    xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
    out = tf.reduce_max(tf.nn.relu(xxc), axis=2)
    return out


def multi_conv1d(in_, filter_sizes, heights, padding="VALID", is_train=True,
                 drop_rate=0.0, scope=None):
  with tf.variable_scope(scope or "multi_conv1d"):
    assert len(filter_sizes) == len(heights)
    outs = []
    for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
      if filter_size == 0:
        continue
      out = conv1d(in_,
                   filter_size,
                   height,
                   padding,
                   is_train=is_train,
                   drop_rate=drop_rate,
                   scope="conv1d_{}".format(i))
      outs.append(out)
    concat_out = tf.concat(axis=2, values=outs)
    return concat_out
