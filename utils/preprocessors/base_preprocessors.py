# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
from collections import Counter
import numpy as np
from tqdm import tqdm

from utils.common import PAD, UNK, NUM, word_convert

glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6),
               '2B': int(1.2e6)}


class Preprocessor(object):

  def __init__(self, config):
    self.config = config

  @staticmethod
  def raw_dataset_iter(filename, keep_number, lowercase):
    with codecs.open(filename, mode="r", encoding="utf-8") as f:
      words, tags = [], []
      for line in f:
        line = line.lstrip().rstrip()
        if line.startswith("-DOCSTART-"):
          continue
        if len(line) == 0:
          if len(words) != 0:
            yield words, tags
            words, tags = [], []
        else:
          line = line.split()
          word = line[0]
          tag = line[-1]
          word = word_convert(word, keep_number=keep_number,
                              lowercase=lowercase)
          words.append(word)
          tags.append(tag)

  def load_dataset(self, filename, keep_number=False, lowercase=True):
    dataset = []
    for words, tags in self.raw_dataset_iter(filename, keep_number, lowercase):
      dataset.append({"words": words, "tags": tags})
    return dataset

  @staticmethod
  def load_glove_vocab(glove_path, glove_name):
    vocab = set()
    total = glove_sizes[glove_name]
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
      for line in tqdm(f, total=total, desc="Load glove vocabulary"):
        line = line.lstrip().rstrip().split(" ")
        vocab.add(line[0])
    return vocab

  @staticmethod
  def build_word_vocab(datasets):
    word_counter = Counter()
    for dataset in datasets:
      for record in dataset:
        words = record["words"]
        for word in words:
          word_counter[word] += 1
    word_vocab = [PAD, UNK, NUM] + [word for word, _ in
                                    word_counter.most_common(10000) if
                                    word != NUM]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict

  @staticmethod
  def build_char_vocab(datasets):
    char_counter = Counter()
    for dataset in datasets:
      for record in dataset:
        for word in record["words"]:
          for char in word:
            char_counter[char] += 1
    word_vocab = [PAD, UNK] + sorted(
      [char for char, _ in char_counter.most_common()])
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict

  @staticmethod
  def build_word_vocab_pretrained(datasets, glove_vocab):
    word_counter = Counter()
    for dataset in datasets:
      for record in dataset:
        words = record["words"]
        for word in words:
          word_counter[word] += 1
    # build word dict
    word_vocab = [PAD, UNK, NUM] + sorted(list(glove_vocab))
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict

  @staticmethod
  def filter_glove_emb(word_dict, glove_path, glove_name, dim):
    vectors = np.zeros([len(word_dict) - 3, dim])
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
      for line in tqdm(f, total=glove_sizes[glove_name],
                       desc="Filter glove embeddings"):
        line = line.lstrip().rstrip().split(" ")
        word = line[0]
        vector = [float(x) for x in line[1:]]
        if word in word_dict:
          word_idx = word_dict[word] - 3
          vectors[word_idx] = np.asarray(vector)
    return vectors

  @staticmethod
  def build_tag_vocab(datasets):
    raise NotImplementedError

  @staticmethod
  def build_dataset(data, word_dict, char_dict, tag_dict):
    raise NotImplementedError

  def preprocess(self):
    raise NotImplementedError
