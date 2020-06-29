# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, MultiRNNCell

from utils import get_logger
from utils.common import load_json, word_convert, UNK


class BaseModel(object):

  def __init__(self, config, batcher, is_train=True):
    self.cfg = config
    self.batcher = batcher
    self.sess = None
    self.saver = None

    self._initialize_config()
    self._add_placeholders()
    self._build_embedding_op()
    self._build_model_op()
    if is_train:
      self._build_loss_op()
      self._build_train_op()
    self._build_predict_op()
    print('Num. params: {}'.format(
      np.sum([np.prod(v.get_shape().as_list())
              for v in tf.trainable_variables()])))
    self.initialize_session()

  def _initialize_config(self):
    # create folders and logger
    os.makedirs(self.cfg["checkpoint_path"], exist_ok=True)
    os.makedirs(os.path.join(self.cfg["summary_path"]), exist_ok=True)
    self.logger = get_logger(
      os.path.join(self.cfg["checkpoint_path"], "log.txt"))

    # load dictionary
    dict_data = load_json(self.cfg["vocab"])
    self.word_dict = dict_data["word_dict"]
    self.char_dict = dict_data["char_dict"]
    self.tag_dict = dict_data["tag_dict"]
    del dict_data
    self.word_vocab_size = len(self.word_dict)
    self.char_vocab_size = len(self.char_dict)
    self.tag_vocab_size = len(self.tag_dict)
    self.rev_word_dict = dict([(idx, word)
                               for word, idx in self.word_dict.items()])
    self.rev_char_dict = dict([(idx, char)
                               for char, idx in self.char_dict.items()])
    self.rev_tag_dict = dict([(idx, tag)
                              for tag, idx in self.tag_dict.items()])

  def initialize_session(self):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=sess_config)
    self.saver = tf.train.Saver(max_to_keep=self.cfg["max_to_keep"])
    self.sess.run(tf.global_variables_initializer())

  def restore_last_session(self, ckpt_path=None):
    if ckpt_path is not None:
      ckpt = tf.train.get_checkpoint_state(ckpt_path)
    else:
      ckpt = tf.train.get_checkpoint_state(self.cfg["checkpoint_path"])
    if ckpt and ckpt.model_checkpoint_path:  # restore session
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)

  def log_trainable_variables(self):
    self.logger.info("\nTrainable variable")
    for v in tf.trainable_variables():
      self.logger.info("-- {}: shape:{}".format(
        v.name, v.get_shape().as_list()))

  def save_session(self, epoch):
    self.saver.save(self.sess,
                    os.path.join(self.cfg["checkpoint_path"],
                                 self.cfg["model_name"]),
                    global_step=epoch)

  def close_session(self):
    self.sess.close()

  def _add_summary(self):
    self.summary = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(
      os.path.join(self.cfg["summary_path"], "train"),
      self.sess.graph)
    self.test_writer = tf.summary.FileWriter(
      os.path.join(self.cfg["summary_path"], "test"))

  def reinitialize_weights(self, scope_name=None):
    """Reinitialize parameters in a scope"""
    if scope_name is None:
      self.sess.run(tf.global_variables_initializer())
    else:
      variables = tf.contrib.framework.get_variables(scope_name)
      self.sess.run(tf.variables_initializer(variables))

  @staticmethod
  def variable_summaries(variable, name=None):
    with tf.name_scope(name or "summary"):
      mean = tf.reduce_mean(variable)
      tf.summary.scalar("mean", mean)  # add mean value
      stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
      tf.summary.scalar("stddev", stddev)  # add standard deviation value
      tf.summary.scalar("max", tf.reduce_max(variable))  # add maximal value
      tf.summary.scalar("min", tf.reduce_min(variable))  # add minimal value
      tf.summary.histogram("histogram", variable)  # add histogram

  def _create_single_rnn_cell(self, num_units):
    return GRUCell(num_units) \
      if self.cfg["cell_type"] == "gru" else LSTMCell(num_units)

  def _create_rnn_cell(self):
    if self.cfg["num_layers"] is None or self.cfg["num_layers"] <= 1:
      return self._create_single_rnn_cell(self.cfg["num_units"])
    else:
      MultiRNNCell([self._create_single_rnn_cell(self.cfg["num_units"])
                    for _ in range(self.cfg["num_layers"])])

  def _add_placeholders(self):
    raise NotImplementedError("To be implemented...")

  def _get_feed_dict(self, data):
    raise NotImplementedError("To be implemented...")

  def _build_embedding_op(self):
    raise NotImplementedError("To be implemented...")

  def _build_model_op(self):
    raise NotImplementedError("To be implemented...")

  def _build_loss_op(self):
    raise NotImplementedError("To be implemented...")

  def _build_train_op(self):
    raise NotImplementedError("To be implemented...")

  def _build_predict_op(self):
    raise NotImplementedError("To be implemented...")

  def train_epoch(self, **kwargs):
    raise NotImplementedError("To be implemented...")

  def train(self, **kwargs):
    raise NotImplementedError("To be implemented...")

  def words_to_indices(self, words):
    chars_idx = []
    for word in words:
      chars = [self.char_dict[char]
               if char in self.char_dict else self.char_dict[UNK]
               for char in word]
      chars_idx.append(chars)
      words = [word_convert(word) for word in words]
      words_idx = [self.word_dict[word]
                   if word in self.word_dict else self.word_dict[UNK]
                   for word in words]
      return self.batcher.make_each_batch([words_idx], [chars_idx])
