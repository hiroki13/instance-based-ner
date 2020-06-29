# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.common import load_json


def metrics_for_multi_class_spans(batch_gold_spans, batch_pred_spans,
                                  null_label_id=0):
  correct = 0
  p_total = 0
  r_total = 0
  for gold_spans, pred_spans in zip(batch_gold_spans, batch_pred_spans):
    assert len(gold_spans) == len(pred_spans)
    r_total += sum([1 for e in gold_spans if e > null_label_id])
    p_total += sum([1 for e in pred_spans if e > null_label_id])
    correct += sum(
      [1 for t, p in zip(gold_spans, pred_spans) if t == p > null_label_id])
  return correct, p_total, r_total


def count_gold_and_system_outputs(batch_gold_spans, batch_pred_spans,
                                  null_label_id=0):
  correct = 0
  p_total = 0
  for gold_spans, pred_spans in zip(batch_gold_spans, batch_pred_spans):
    assert len(gold_spans) == len(pred_spans)
    p_total += sum([1 for e in pred_spans if e > null_label_id])
    correct += sum(
      [1 for t, p in zip(gold_spans, pred_spans) if t == p > null_label_id])
  return correct, p_total


def count_gold_spans(path):
  n_gold_spans = 0
  for record in load_json(path):
    n_gold_spans += len(record["tags"])
  return n_gold_spans


def f_score(correct, p_total, r_total):
  precision = correct / p_total if p_total > 0 else 0.
  recall = correct / r_total if r_total > 0 else 0.
  f1 = (2 * precision * recall) / (
    precision + recall) if precision + recall > 0 else 0.
  return precision, recall, f1


def align_data(data):
  """Given dict with lists, creates aligned strings
  Args:
      data: (dict) data["x"] = ["I", "love", "you"]
            (dict) data["y"] = ["O", "O", "O"]
  Returns:
      data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "
  """
  spacings = [max([len(seq[i]) for seq in data.values()])
              for i in range(len(data[list(data.keys())[0]]))]
  data_aligned = dict()
  # for each entry, create aligned string
  for key, seq in data.items():
    str_aligned = ''
    for token, spacing in zip(seq, spacings):
      str_aligned += token + ' ' * (spacing - len(token) + 1)
    data_aligned[key] = str_aligned
  return data_aligned


def span2bio(spans, n_words, tag_dict=None):
  bio_tags = ['O' for _ in range(n_words)]
  for (label_id, pre_index, post_index) in spans:
    if tag_dict:
      label = tag_dict[label_id]
    else:
      label = str(label_id)
    bio_tags[pre_index] = 'B-%s' % label
    for index in range(pre_index + 1, post_index + 1):
      bio_tags[index] = 'I-%s' % label
  return bio_tags


def bio2triple(tags):
  triples = []
  for i, tag in enumerate(tags):
    if tag.startswith('B-'):
      label = tag[2:]
      triples.append([label, i, i])
    elif tag.startswith('I-'):
      triples[-1] = triples[-1][:-1] + [i]
  return triples


def bio_to_span(datasets):
  for dataset in datasets:
    for record in dataset:
      bio_tags = record["tags"]
      triples = bio2triple(bio_tags)
      record["tags"] = triples
  return datasets
