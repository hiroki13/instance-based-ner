# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import json

from models.knn_models import KnnModel
from utils.batchers.knn_batchers import BaseKnnBatcher
from utils.preprocessors.knn_preprocessors import KnnPreprocessor


def set_config(args, config):
  if args.raw_path:
    config["raw_path"] = args.raw_path
  if args.save_path:
    config["save_path"] = args.save_path
    config["train_set"] = os.path.join(args.save_path, "train.json")
    config["valid_set"] = os.path.join(args.save_path, "valid.json")
    config["vocab"] = os.path.join(args.save_path, "vocab.json")
    config["pretrained_emb"] = os.path.join(args.save_path, "glove_emb.npz")
  if args.train_set:
    config["train_set"] = args.train_set
  if args.valid_set:
    config["valid_set"] = args.valid_set
  if args.pretrained_emb:
    config["pretrained_emb"] = args.pretrained_emb
  if args.vocab:
    config["vocab"] = args.vocab
  if args.checkpoint_path:
    config["checkpoint_path"] = args.checkpoint_path
    config["summary_path"] = os.path.join(args.checkpoint_path, "summary")
  if args.summary_path:
    config["summary_path"] = args.summary_path
  if args.model_name:
    config["model_name"] = args.model_name
  if args.batch_size:
    config["batch_size"] = args.batch_size
  if args.data_size:
    config["data_size"] = args.data_size
  if args.bilstm_type:
    config["bilstm_type"] = args.bilstm_type
  if args.keep_prob:
    config["keep_prob"] = args.keep_prob
  if args.k:
    config["k"] = args.k
  if args.predict:
    config["predict"] = args.predict
  if args.max_span_len:
    config["max_span_len"] = args.max_span_len
  if args.max_n_spans:
    config["max_n_spans"] = args.max_n_spans
  if args.knn_sampling:
    config["knn_sampling"] = args.knn_sampling
  return config


def main(args):
  config = json.load(open(args.config_file))
  config = set_config(args, config)

  preprocessor = KnnPreprocessor(config)

  # create dataset from raw data files
  if not os.path.exists(config["save_path"]):
    preprocessor.preprocess()

  model = KnnModel(config, BaseKnnBatcher(config))
  model.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file',
                      required=True,
                      default='data/config/config.json',
                      help='Configuration file')
  parser.add_argument('--raw_path',
                      default=None,
                      help='Raw data directory')
  parser.add_argument('--save_path',
                      default=None,
                      help='Save directory')
  parser.add_argument('--checkpoint_path',
                      default=None,
                      help='Checkpoint directory')
  parser.add_argument('--summary_path',
                      default=None,
                      help='Summary directory')
  parser.add_argument('--model_name',
                      default=None,
                      help='Model name')
  parser.add_argument('--batch_size',
                      default=None,
                      type=int,
                      help='Batch size')
  parser.add_argument('--train_set',
                      default=None,
                      help='path to training set')
  parser.add_argument('--valid_set',
                      default=None,
                      help='path to training set')
  parser.add_argument('--pretrained_emb',
                      default=None,
                      help='path to pretrained embeddings')
  parser.add_argument('--vocab',
                      default=None,
                      help='path to vocabulary')
  parser.add_argument('--data_size',
                      default=None,
                      type=int,
                      help='Data size')
  parser.add_argument('--bilstm_type',
                      default=None,
                      help='standard/interleave')
  parser.add_argument('--keep_prob',
                      default=None,
                      type=float,
                      help='Keep (dropout) probability')
  parser.add_argument('--k',
                      default=None,
                      type=int,
                      help='k-NN sentences')
  parser.add_argument('--predict',
                      default='max_margin',
                      help='prediction methods')
  parser.add_argument('--max_span_len',
                      default=None,
                      type=int,
                      help='max span length')
  parser.add_argument('--max_n_spans',
                      default=None,
                      type=int,
                      help='max num of spans')
  parser.add_argument('--knn_sampling',
                      default=None,
                      help='k-NN sentence sampling')
  main(parser.parse_args())
