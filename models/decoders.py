from typing import List
import numpy as np


def get_span_indices(n_words, max_span_len):
  ones = np.ones(shape=(n_words, n_words), dtype='int32')
  mask = np.triu(ones, k=np.minimum(n_words, max_span_len))
  indices = np.triu(ones) - mask
  return np.nonzero(indices)


def get_scored_spans(scores, n_words, max_span_len) -> List[List[tuple]]:
  indx_i, indx_j = get_span_indices(n_words, max_span_len)
  assert len(scores) == len(indx_i) == len(indx_j), "%d %d %d" % (len(scores),
                                                                  len(indx_i),
                                                                  len(indx_j))
  spans = []
  for scores_each_span, i, j in zip(scores, indx_i, indx_j):
    spans.append(
      [(r, i, j, score) for r, score in enumerate(scores_each_span)])
  return spans


def get_labeled_spans(labels, n_words, max_span_len,
                      null_label_id) -> List[tuple]:
  indx_i, indx_j = get_span_indices(n_words, max_span_len)
  assert len(labels) == len(indx_i) == len(indx_j), "%d %d %d" % (len(labels),
                                                                  len(indx_i),
                                                                  len(indx_j))
  return [(r, i, j) for r, i, j in zip(labels, indx_i, indx_j)
          if r != null_label_id]


def get_scores_and_spans(spans, scores, sent_id, indx_i, indx_j) -> List[List]:
  scored_spans = []
  span_boundaries = [(i, j) for (_, i, j) in spans]
  for i, j, score in zip(indx_i, indx_j, scores):
    if (i, j) in span_boundaries:
      index = span_boundaries.index((i, j))
      r = spans[index][0]
    else:
      r = "O"
    scored_spans.append([r, sent_id, i, j, score])
  return scored_spans


def get_batch_labeled_spans(batch_labels, n_words, max_span_len,
                            null_label_id) -> List[List[List[int]]]:
  indx_i, indx_j = get_span_indices(n_words, max_span_len)
  return [[[r, i, j] for r, i, j in zip(labels, indx_i, indx_j) if
           r != null_label_id]
          for labels in batch_labels]


def get_pred_spans_with_proba(scores, n_words, max_span_len) -> dict:
  indx_i, indx_j = get_span_indices(n_words, max_span_len)
  assert len(scores) == len(indx_i) == len(indx_j), "%d %d %d" % (len(scores),
                                                                  len(indx_i),
                                                                  len(indx_j))
  spans = {}
  for label_scores, i, j in zip(scores, indx_i, indx_j):
    spans['%d,%d' % (i, j)] = [float(score) for score in label_scores]
  return spans


def sort_scored_spans(scored_spans, null_label_id) -> List[tuple]:
  sorted_spans = []
  for spans in scored_spans:
    r, i, j, null_score = spans[null_label_id]
    for (r, i, j, score) in spans:
      if null_score < score:
        sorted_spans.append((r, i, j, score))
  sorted_spans.sort(key=lambda span: span[-1], reverse=True)
  return sorted_spans


def greedy_search(scores, n_words, max_span_len,
                  null_label_id) -> List[List[int]]:
  triples = []
  used_words = np.zeros(n_words, 'int32')
  scored_spans = get_scored_spans(scores, n_words, max_span_len)
  sorted_spans = sort_scored_spans(scored_spans, null_label_id)

  for (r, i, j, _) in sorted_spans:
    if sum(used_words[i: j + 1]) > 0:
      continue
    triples.append([r, i, j])
    used_words[i: j + 1] = 1
  return triples
