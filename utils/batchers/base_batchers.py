# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class Batcher(object):

  def __init__(self, config):
    self.config = config

  @staticmethod
  def pad_sequences(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
      # 0: "PAD" for words and chars, "O" for tags
      pad_tok = 0
    if max_length is None:
      max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
      seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
      sequence_padded.append(seq_)
      sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length

  def pad_char_sequences(self, sequences, max_length=None,
                         max_token_length=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
      max_length = max(map(lambda x: len(x), sequences))
    if max_token_length is None:
      max_token_length = max(
        [max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
      sp, sl = self.pad_sequences(seq, max_length=max_token_length)
      sequence_padded.append(sp)
      sequence_length.append(sl)
    sequence_padded, _ = self.pad_sequences(sequence_padded,
                                            pad_tok=[0] * max_token_length,
                                            max_length=max_length)
    sequence_length, _ = self.pad_sequences(sequence_length,
                                            max_length=max_length)
    return sequence_padded, sequence_length

  def make_each_batch(self, **kwargs):
    raise NotImplementedError

  def batchnize_dataset(self, **kwargs):
    raise NotImplementedError
