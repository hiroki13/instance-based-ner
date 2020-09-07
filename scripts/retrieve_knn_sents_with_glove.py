# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import ujson
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6)}


def load_json(filename):
  with codecs.open(filename, mode='r', encoding='utf-8') as f:
    dataset = ujson.load(f)
  return dataset


def write_json(filename, data):
  with codecs.open(filename, mode="w", encoding="utf-8") as f:
    ujson.dump(data, f, ensure_ascii=False)


def load_glove(glove_path, glove_name="6B"):
  vocab = {}
  vectors = []
  total = glove_sizes[glove_name]
  with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
    for line in tqdm(f, total=total, desc="Load glove"):
      line = line.lstrip().rstrip().split(" ")
      vocab[line[0]] = len(vocab)
      vectors.append([float(x) for x in line[1:]])
  assert len(vocab) == len(vectors)
  return vocab, np.asarray(vectors)


def mean_vectors(data, emb, vocab):
  unk_vec = np.zeros(emb.shape[1])
  mean_vecs = []
  for record in data:
    vecs = []
    for word in record["words"]:
      word = word.lower()
      if word in vocab:
        vec = emb[vocab[word]]
      else:
        vec = unk_vec
      vecs.append(vec)
    mean_vecs.append(np.mean(vecs, axis=0))
  return mean_vecs


def cosine_similarity(p0, p1):
  d = (norm(p0) * norm(p1))
  if d > 0:
    return np.dot(p0, p1) / d
  return 0.0


def knn(test_sents, train_embs, test_embs, k, path):
  for index, (sent, vec) in enumerate(zip(test_sents, test_embs)):
    assert index == sent["sent_id"]
    if (index + 1) % 100 == 0:
      print("%d" % (index + 1), flush=True, end=" ")
    sim = [cosine_similarity(train_vec, vec) for train_vec in train_embs]
    arg_sort = np.argsort(sim)[::-1][:k]
    sent["train_sent_ids"] = [int(arg) for arg in arg_sort]
  write_json(path, test_sents)


def main(args):
  train_sents = load_json(args.train_json)[:args.data_size]
  test_sents = load_json(args.test_json)[:args.data_size]
  vocab, glove = load_glove(args.glove)
  print("Train sents: {:>7}".format(len(train_sents)))
  print("Test  sents: {:>7}".format(len(test_sents)))
  train_embs = mean_vectors(train_sents, glove, vocab)
  test_embs = mean_vectors(test_sents, glove, vocab)
  if args.output_file.endswith(".json"):
    path = args.output_file
  else:
    path = args.output_file + ".json"
  knn(test_sents, train_embs, test_embs, args.k, path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='SCRIPT')
  parser.add_argument('--train_json',
                      type=str,
                      default='data/conll2003/train.json',
                      help='path to json-format data')
  parser.add_argument('--test_json',
                      type=str,
                      default='data/conll2003/test.json',
                      help='path to json-format data')
  parser.add_argument('--output_file',
                      default="output",
                      help='output file name')
  parser.add_argument('--glove',
                      type=str,
                      default='data/emb/glove.6B.100d.txt',
                      help='path to glove embeddings')
  parser.add_argument('--k',
                      type=int,
                      default=50,
                      help='k')
  parser.add_argument('--data_size',
                      type=int,
                      default=100000000,
                      help='number of sentences to be used')
  main(parser.parse_args())
