# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

import numpy as np

from utils.common import load_json
from utils.batchers.base_batchers import Batcher


class SpanBatcher(Batcher):

  @staticmethod
  def _make_span_indices(n_words, max_span_len, max_n_spans=0):
    indices = np.triu(np.ones(shape=(n_words, n_words), dtype='int32'))
    mask = np.triu(np.ones(shape=(n_words, n_words), dtype='int32'),
                   k=max_span_len)
    indices = np.nonzero(indices - mask)

    if max_n_spans > 0:
      indices = [(i, j) for i, j in zip(*indices)]
      random.shuffle(indices)
      return np.asarray(list(zip(*indices[:max_n_spans])), dtype='int32')
    return indices

  @staticmethod
  def _make_tag_sequences(batch_triples, indices, n_words):
    def _gen_tag_sequence(triples):
      matrix = np.zeros(shape=(n_words, n_words), dtype='int32')
      for (r, i, j) in triples:
        matrix[i][j] = r
      return [matrix[i][j] for (i, j) in zip(*indices)]

    return np.asarray([_gen_tag_sequence(triples)
                       for triples in batch_triples], dtype='int32')

  def make_each_batch(self, **kwargs):
    raise NotImplementedError

  def batchnize_dataset(self, **kwargs):
    raise NotImplementedError


class BaseSpanBatcher(SpanBatcher):

  def make_each_batch(self, batch_words, batch_chars, max_span_len=None,
                      batch_tags=None):
    b_words, b_words_len = self.pad_sequences(batch_words)
    b_chars, _ = self.pad_char_sequences(batch_chars, max_token_length=20)
    n_words = b_words_len[0]
    span_indices = self._make_span_indices(n_words, max_span_len)
    if batch_tags is None:
      return {"words": b_words,
              "chars": b_chars,
              "seq_len": b_words_len}
    else:
      b_tags = self._make_tag_sequences(batch_tags, span_indices, n_words)
      return {"words": b_words,
              "chars": b_chars,
              "tags": b_tags,
              "seq_len": b_words_len}

  def batchnize_dataset(self, data, batch_size=None, shuffle=True):
    batches = []
    max_span_len = self.config["max_span_len"]
    dataset = load_json(data)

    if shuffle:
      random.shuffle(dataset)
      dataset.sort(key=lambda record: len(record["words"]))

    prev_seq_len = len(dataset[0]["words"])
    batch_words, batch_chars, batch_tags = [], [], []

    for record in dataset:
      seq_len = len(record["words"])

      if len(batch_words) == batch_size or prev_seq_len != seq_len:
        batches.append(self.make_each_batch(batch_words,
                                            batch_chars,
                                            max_span_len,
                                            batch_tags))
        batch_words, batch_chars, batch_tags = [], [], []
        prev_seq_len = seq_len

      batch_words.append(record["words"])
      batch_chars.append(record["chars"])
      batch_tags.append(record["tags"])

    if len(batch_words) > 0:
      batches.append(self.make_each_batch(batch_words,
                                          batch_chars,
                                          max_span_len,
                                          batch_tags))
    if shuffle:
      random.shuffle(batches)
    for batch in batches:
      yield batch


class MaskSpanBatcher(SpanBatcher):

  @staticmethod
  def _make_masks(lengths, indices, max_n_words):
    def _gen_mask(n_words):
      matrix = np.zeros(shape=(max_n_words, max_n_words), dtype='float32')
      matrix[:n_words, :n_words] = 1.0
      return [matrix[i][j] for (i, j) in zip(*indices)]

    return np.asarray([_gen_mask(n_words) for n_words in lengths],
                      dtype='float32')

  def make_each_batch(self, batch_words, batch_chars, max_span_len=None,
                      max_n_spans=None, batch_tags=None):
    b_words, b_words_len = self.pad_sequences(batch_words)
    b_chars, _ = self.pad_char_sequences(batch_chars, max_token_length=20)
    max_n_words = max(b_words_len)
    span_indices = self._make_span_indices(max_n_words, max_span_len)
    b_masks = self._make_masks(b_words_len, span_indices, max_n_words)
    if batch_tags is None:
      return {"words": b_words,
              "chars": b_chars,
              "masks": b_masks,
              "seq_len": b_words_len}
    else:
      b_tags = self._make_tag_sequences(batch_tags, span_indices, max_n_words)
      return {"words": b_words,
              "chars": b_chars,
              "tags": b_tags,
              "masks": b_masks,
              "seq_len": b_words_len}

  def batchnize_dataset(self, data, batch_size=None, shuffle=True):
    max_span_len = self.config["max_span_len"]
    max_n_spans = None
    dataset = load_json(data)

    if shuffle:
      random.shuffle(dataset)

    batch_words, batch_chars, batch_tags = [], [], []

    for record in dataset:
      if len(batch_words) == batch_size:
        yield self.make_each_batch(batch_words, batch_chars, max_span_len,
                                   max_n_spans, batch_tags)
        batch_words, batch_chars, batch_tags = [], [], []

      batch_words.append(record["words"])
      batch_chars.append(record["chars"])
      batch_tags.append(record["tags"])

    if len(batch_words) > 0:
      yield self.make_each_batch(batch_words, batch_chars, max_span_len,
                                 max_n_spans, batch_tags)
