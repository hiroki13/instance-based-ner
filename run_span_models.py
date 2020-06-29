import argparse
import os
import json

from models.span_models import SpanModel
from utils.batchers.span_batchers import BaseSpanBatcher
from utils.preprocessors.span_preprocessors import SpanPreprocessor


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
    if args.max_span_len:
        config["max_span_len"] = args.max_span_len
    return config


def main(args):
    config = json.load(open(args.config_file))
    config = set_config(args, config)
    os.makedirs(config["save_path"], exist_ok=True)

    print("Build models...")
    model = SpanModel(config, BaseSpanBatcher(config), is_train=False)
    preprocessor = SpanPreprocessor(config)
    model.restore_last_session(config["checkpoint_path"])

    if args.mode == "eval":
        model.eval(preprocessor)
    elif args.mode == "span":
        model.save_predicted_spans(args.data_name, preprocessor)
    elif args.mode == "bio":
        model.build_proba_op()
        model.save_predicted_bio_tags(args.data_name, preprocessor)
    else:
      model.save_span_representation(args.data_name, preprocessor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        default="eval",
                        help='eval/span/bio/proba/span_rep')
    parser.add_argument('--config_file',
                        default='checkpoint/config.json',
                        help='Configuration file')
    parser.add_argument('--data_name',
                        default="valid",
                        help='Data to be processed')
    parser.add_argument('--data_path',
                        required=True,
                        default=None,
                        help='Path to data')
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
    parser.add_argument('--bilstm_type',
                        default=None,
                        help='standard/interleave')
    parser.add_argument('--max_span_len',
                        default=None,
                        type=int,
                        help='max span length')
    main(parser.parse_args())
