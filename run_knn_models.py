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
  if args.data_path:
    config["data_path"] = args.data_path
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
  if args.k:
    config["k"] = args.k
  if args.predict:
    config["predict"] = args.predict
  if args.max_span_len:
    config["max_span_len"] = args.max_span_len
  if args.knn_sampling:
    config["knn_sampling"] = args.knn_sampling
  return config


def main(args):
  config = json.load(open(args.config_file))
  config = set_config(args, config)
  os.makedirs(config["save_path"], exist_ok=True)

  print("Build a knn span model...")
  model = KnnModel(config, BaseKnnBatcher(config), is_train=False)
  preprocessor = KnnPreprocessor(config)
  model.restore_last_session(config["checkpoint_path"])

  if args.mode == "eval":
    model.eval(preprocessor)
  elif args.mode == "span":
    model.save_predicted_spans(args.data_name, preprocessor)
  elif args.mode == "bio":
    model.save_predicted_bio_tags(args.data_name, preprocessor)
  elif args.mode == "nearest_span":
    model.save_nearest_spans(args.data_name, preprocessor, args.print_knn)
  elif args.mode == "cmd":
    model.predict_on_command_line(preprocessor)
  else:
    model.save_span_representation(args.data_name, preprocessor)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode',
                      default='eval',
                      help='eval/span/bio/nearest_span/span_rep')
  parser.add_argument('--config_file',
                      default='checkpoint/config.json',
                      help='Configuration file')
  parser.add_argument('--data_name',
                      default='valid',
                      help='Data to be processed')
  parser.add_argument('--data_path',
                      default=None,
                      help='Path to data')
  parser.add_argument('--bilstm_type',
                      default=None,
                      help='bilstm type')
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
  parser.add_argument('--data_size',
                      default=None,
                      type=int,
                      help='Data size')
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
  parser.add_argument('--knn_sampling',
                      default=None,
                      help='k-NN sentence sampling')
  parser.add_argument('--print_knn',
                      action='store_true',
                      default=False,
                      help='print knn sentences')
  main(parser.parse_args())
