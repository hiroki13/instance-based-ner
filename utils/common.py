# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import gzip
import re
import pickle
import ujson
import unicodedata

PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
SPACE = "_SPACE"
BOS = "<S>"
EOS = "</S>"


def load_json(filename):
  with codecs.open(filename, mode='r', encoding='utf-8') as f:
    dataset = ujson.load(f)
  return dataset


def write_json(filename, data):
  with codecs.open(filename, mode="w", encoding="utf-8") as f:
    ujson.dump(data, f, ensure_ascii=False)


def load_pickle(filename):
  with gzip.open(filename, 'rb') as gf:
    return pickle.load(gf)


def write_pickle(filename, data):
  with gzip.open(filename + '.pkl.gz', 'wb') as gf:
    pickle.dump(data, gf, pickle.HIGHEST_PROTOCOL)


def word_convert(word, keep_number=True, lowercase=True):
  if not keep_number:
    if is_digit(word):
      return NUM
  if lowercase:
    word = word.lower()
  return word


def is_digit(word):
  try:
    float(word)
    return True
  except ValueError:
    pass
  try:
    unicodedata.numeric(word)
    return True
  except (TypeError, ValueError):
    pass
  result = re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(word)
  if result:
    return True
  return False
