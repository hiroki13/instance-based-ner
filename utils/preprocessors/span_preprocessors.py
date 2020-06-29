# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from collections import Counter
import numpy as np

from utils.common import write_json, load_json, UNK, word_convert
from utils.preprocessors.base_preprocessors import Preprocessor


class SpanPreprocessor(Preprocessor):

  def load_dataset(self, filename, keep_number=False, lowercase=True):
    dataset = []
    for record in load_json(filename):
      words = [word_convert(word, keep_number=keep_number, lowercase=lowercase)
               for word in record["words"]]
      dataset.append({"sent_id": record["sent_id"],
                      "words": words,
                      "tags": record["spans"]})
    return dataset

  @staticmethod
  def build_tag_vocab(datasets):
    tag_counter = Counter()
    for dataset in datasets:
      for record in dataset:
        for (tag, _, _) in record["tags"]:
          tag_counter[tag] += 1
    tag_vocab = ["O"] + [tag for tag, _ in tag_counter.most_common()]
    tag_dict = dict([(ner, idx) for idx, ner in enumerate(tag_vocab)])
    return tag_dict

  @staticmethod
  def build_dataset(data, word_dict, char_dict, tag_dict):
    dataset = []
    for record in data:
      chars_list = []
      words = []
      for word in record["words"]:
        chars = [char_dict[char] if char in char_dict else char_dict[UNK] for
                 char in word]
        chars_list.append(chars)
        word = word_convert(word, keep_number=False, lowercase=True)
        words.append(word_dict[word] if word in word_dict else word_dict[UNK])
      tags = [(tag_dict[tag], i, j) for (tag, i, j) in record["tags"]]
      dataset.append({"words": words, "chars": chars_list, "tags": tags})
    return dataset

  def preprocess(self):
    config = self.config
    os.makedirs(config["save_path"], exist_ok=True)

    # List[{'words': List[str], 'tags': List[str]}]
    train_data = self.load_dataset(
      os.path.join(config["raw_path"], "train.json"),
      keep_number=False,
      lowercase=True)
    valid_data = self.load_dataset(
      os.path.join(config["raw_path"], "valid.json"),
      keep_number=False,
      lowercase=True)
    train_data = train_data[:config["data_size"]]
    valid_data = valid_data[:config["data_size"]]

    # build vocabulary
    if config["use_pretrained"]:
      glove_path = self.config["glove_path"].format(config["glove_name"],
                                                    config["emb_dim"])
      glove_vocab = self.load_glove_vocab(glove_path, config["glove_name"])
      word_dict = self.build_word_vocab_pretrained([train_data, valid_data],
                                                   glove_vocab)
      vectors = self.filter_glove_emb(word_dict,
                                      glove_path,
                                      config["glove_name"],
                                      config["emb_dim"])
      np.savez_compressed(config["pretrained_emb"], embeddings=vectors)
    else:
      word_dict = self.build_word_vocab([train_data, valid_data])

    # build tag dict
    tag_dict = self.build_tag_vocab([train_data, valid_data])

    # build char dict
    train_data = self.load_dataset(
      os.path.join(config["raw_path"], "train.json"),
      keep_number=True,
      lowercase=config["char_lowercase"])
    valid_data = self.load_dataset(
      os.path.join(config["raw_path"], "valid.json"),
      keep_number=True,
      lowercase=config["char_lowercase"])

    train_data = train_data[:config["data_size"]]
    valid_data = valid_data[:config["data_size"]]

    if config["max_sent_len"] > 0:
      train_data = [record for record in train_data
                    if len(record["words"]) <= config["max_sent_len"]]
      print(
        "Train Sents (remove max_sent_len: %d): %d" % (config["max_sent_len"],
                                                       len(train_data)))

    char_dict = self.build_char_vocab([train_data])

    # create indices dataset
    # List[{'words': List[str], 'chars': List[List[str]], 'tags': List[str]}]
    train_set = self.build_dataset(train_data, word_dict, char_dict, tag_dict)
    valid_set = self.build_dataset(valid_data, word_dict, char_dict, tag_dict)
    vocab = {"word_dict": word_dict,
             "char_dict": char_dict,
             "tag_dict": tag_dict}

    print("Train Sents: %d" % len(train_set))
    print("Valid Sents: %d" % len(valid_set))

    # write to file
    write_json(os.path.join(config["save_path"], "vocab.json"), vocab)
    write_json(os.path.join(config["save_path"], "train.json"), train_set)
    write_json(os.path.join(config["save_path"], "valid.json"), valid_set)
