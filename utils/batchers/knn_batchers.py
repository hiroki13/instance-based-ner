# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import numpy as np

from utils.batchers.span_batchers import MaskSpanBatcher
from utils.common import load_json


class KnnBatcher(MaskSpanBatcher):

  def make_each_batch_for_targets(self, **kwargs):
    raise NotImplementedError

  def make_each_batch_for_neighbors(self, **kwargs):
    raise NotImplementedError

  def batchnize_neighbor_train_sents(self, **kwargs):
    raise NotImplementedError

  def make_batch_from_sent_ids(self, **kwargs):
    raise NotImplementedError

  def batchnize_span_reps_and_tags(self, train_batch_reps, train_batch_tags,
                                   train_batch_masks):
    """
    :param train_batch_reps: 1D: num_sents, 2D: max_num_spans, 3D: dim
    :param train_batch_tags: 1D: num_sents, 2D: max_num_spans
    :param train_batch_masks: 1D: num_sents, 2D: max_num_spans
    :return: batch_span_rep_list: 1D: num_instances, 2D: dim
    :return: batch_tag_one_hot_list: 1D: num_instances, 2D: num_tags
    """
    rep_list = []
    tag_list = []
    for reps, tags, masks in zip(train_batch_reps,
                                 train_batch_tags,
                                 train_batch_masks):
      for rep, tag, mask in zip(reps, tags, masks):
        if mask:
          rep_list.append(rep)
          tag_list.append(tag)
    return rep_list, tag_list

  @staticmethod
  def batchnize_span_reps_and_tag_one_hots(train_batch_reps,
                                           train_batch_tags,
                                           train_batch_masks,
                                           num_tags):
    """
    :param train_batch_reps: 1D: num_sents, 2D: max_num_spans, 3D: dim
    :param train_batch_tags: 1D: num_sents, 2D: max_num_spans
    :param train_batch_masks: 1D: num_sents, 2D: max_num_spans
    :param num_tags; the number of possible tags
    :return: batch_span_rep_list: 1D: num_instances, 2D: dim
    :return: batch_tag_one_hot_list: 1D: num_instances, 2D: num_tags
    """
    rep_list = []
    tag_list = []
    for reps, tags, masks in zip(train_batch_reps,
                                 train_batch_tags,
                                 train_batch_masks):
      for rep, tag, mask in zip(reps, tags, masks):
        if mask:
          rep_list.append(rep)
          tag_list.append(tag)

    tag_one_hot_list = np.zeros(shape=(len(tag_list), num_tags))
    for i, tag_id in enumerate(tag_list):
      tag_one_hot_list[i][tag_id] = 1

    return rep_list, tag_one_hot_list


class BaseKnnBatcher(KnnBatcher):

  def make_each_batch_for_targets(self, batch_words, batch_chars, batch_ids,
                                  max_span_len, max_n_spans, batch_tags=None):
    b_words, b_words_len = self.pad_sequences(batch_words)
    b_chars, _ = self.pad_char_sequences(batch_chars, max_token_length=20)
    batch = {"words": b_words,
             "chars": b_chars,
             "seq_len": b_words_len,
             "instance_ids": batch_ids}
    n_words = b_words_len[0]
    span_indices = self._make_span_indices(n_words=n_words,
                                           max_span_len=max_span_len,
                                           max_n_spans=max_n_spans)
    if max_n_spans:
      batch["span_indices"] = span_indices
    if batch_tags is not None:
      batch["tags"] = self._make_tag_sequences(batch_triples=batch_tags,
                                               indices=span_indices,
                                               n_words=n_words)
    return batch

  def make_each_batch_for_neighbors(self, batch_words, batch_chars,
                                    max_span_len, max_n_spans,
                                    batch_tags=None):
    b_words, b_words_len = self.pad_sequences(batch_words)
    b_chars, _ = self.pad_char_sequences(batch_chars, max_token_length=20)
    max_n_words = max(b_words_len)
    span_indices = self._make_span_indices(n_words=max_n_words,
                                           max_span_len=max_span_len,
                                           max_n_spans=max_n_spans)
    b_masks = self._make_masks(lengths=b_words_len,
                               indices=span_indices,
                               max_n_words=max_n_words)
    batch = {"words": b_words,
             "chars": b_chars,
             "seq_len": b_words_len,
             "masks": b_masks}
    if max_n_spans:
      batch["span_indices"] = span_indices
    if batch_tags is not None:
      batch["tags"] = self._make_tag_sequences(batch_triples=batch_tags,
                                               indices=span_indices,
                                               n_words=max_n_words)
    return batch

  def batchnize_dataset(self, data, data_name=None, batch_size=None,
                        shuffle=True):
    max_span_len = self.config["max_span_len"]
    if data_name == "train":
      max_n_spans = self.config["max_n_spans"]
    else:
      if self.config["max_n_spans"] > 0:
        max_n_spans = 1000000
      else:
        max_n_spans = 0

    dataset = load_json(data)
    for instance_id, record in enumerate(dataset):
      record["instance_id"] = instance_id

    if shuffle:
      random.shuffle(dataset)
      dataset.sort(key=lambda record: len(record["words"]))

    batches = []
    batch_words, batch_chars, batch_tags, batch_ids = [], [], [], []
    prev_seq_len = len(dataset[0]["words"])

    for record in dataset:
      seq_len = len(record["words"])

      if len(batch_words) == batch_size or prev_seq_len != seq_len:
        batches.append(self.make_each_batch_for_targets(batch_words,
                                                        batch_chars,
                                                        batch_ids,
                                                        max_span_len,
                                                        max_n_spans,
                                                        batch_tags))
        batch_words, batch_chars, batch_tags, batch_ids = [], [], [], []
        prev_seq_len = seq_len

      batch_words.append(record["words"])
      batch_chars.append(record["chars"])
      batch_tags.append(record["tags"])
      batch_ids.append(record["instance_id"])

    if len(batch_words) > 0:
      batches.append(self.make_each_batch_for_targets(batch_words,
                                                      batch_chars,
                                                      batch_ids,
                                                      max_span_len,
                                                      max_n_spans,
                                                      batch_tags))
    if shuffle:
      random.shuffle(batches)
    for batch in batches:
      yield batch

  def batchnize_neighbor_train_sents(self, train_sents, train_sent_ids,
                                     max_span_len, max_n_spans):
    batch_words, batch_chars, batch_tags = [], [], []
    for sent_id in train_sent_ids:
      batch_words.append(train_sents[sent_id]["words"])
      batch_chars.append(train_sents[sent_id]["chars"])
      batch_tags.append(train_sents[sent_id]["tags"])
    return self.make_each_batch_for_neighbors(batch_words,
                                              batch_chars,
                                              max_span_len,
                                              max_n_spans,
                                              batch_tags)

  def make_batch_from_sent_ids(self, train_sents, sent_ids):
    batch_words, batch_chars, batch_tags = [], [], []
    max_n_spans = 0
    for sent_id in sent_ids:
      batch_words.append(train_sents[sent_id]["words"])
      batch_chars.append(train_sents[sent_id]["chars"])
      batch_tags.append(train_sents[sent_id]["tags"])
    return self.make_each_batch_for_neighbors(batch_words,
                                              batch_chars,
                                              self.config["max_span_len"],
                                              max_n_spans,
                                              batch_tags)
