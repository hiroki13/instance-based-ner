# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import os
import math
import random
import time

import h5py
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tqdm import tqdm

from models.decoders import get_scores_and_spans, get_span_indices, \
  greedy_search
from models.span_models import MaskSpanModel
from utils.common import load_json, write_json, word_convert, UNK
from utils.data_utils import f_score, count_gold_spans, \
  count_gold_and_system_outputs, span2bio

NULL_LABEL_ID = 0


class KnnModel(MaskSpanModel):

  def __init__(self, config, batcher, is_train=True):
    self.knn_ids = None
    self.gold_label_proba = None
    self.max_n_spans = config["max_n_spans"]
    super(KnnModel, self).__init__(config, batcher, is_train)

  def _add_placeholders(self):
    self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
    self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")
    self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
    self.neighbor_reps = tf.placeholder(tf.float32, shape=[None, None],
                                        name="neighbor_reps")
    self.neighbor_tags = tf.placeholder(tf.float32, shape=[None],
                                        name="neighbor_tags")
    self.neighbor_tag_one_hots = tf.placeholder(tf.float32, shape=[None, None],
                                                name="neighbor_tag_one_hots")
    if self.cfg["use_chars"]:
      self.chars = tf.placeholder(tf.int32, shape=[None, None, None],
                                  name="chars")
    # hyperparameters
    self.is_train = tf.placeholder(tf.bool, name="is_train")
    self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
    self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.lr = tf.placeholder(tf.float32, name="learning_rate")

  def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
    feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"]}
    if "tags" in batch:
      feed_dict[self.tags] = batch["tags"]
    if self.cfg["use_chars"]:
      feed_dict[self.chars] = batch["chars"]
    if "neighbor_reps" in batch:
      feed_dict[self.neighbor_reps] = batch["neighbor_reps"]
    if "neighbor_tags" in batch:
      feed_dict[self.neighbor_tags] = batch["neighbor_tags"]
    if "neighbor_tag_one_hots" in batch:
      feed_dict[self.neighbor_tag_one_hots] = batch["neighbor_tag_one_hots"]
    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr
    return feed_dict

  def _build_neighbor_similarity_op(self):
    with tf.name_scope("similarity"):
      # 1D: batch_size, 2D: max_num_spans, 3D: num_instances
      self.similarity = tf.tensordot(self.span_rep, self.neighbor_reps,
                                     axes=[-1, -1])

  def _build_neighbor_proba_op(self):
    with tf.name_scope("neighbor_prob"):
      # 1D: batch_size, 2D: max_num_spans, 3D: num_instances
      self.neighbor_proba = tf.nn.softmax(self.similarity, axis=-1)

  def _build_marginal_proba_op(self):
    with tf.name_scope("gold_label_prob"):
      # 1D: batch_size, 2D: max_num_spans, 3D: 1
      tags = tf.expand_dims(tf.cast(self.tags, dtype=tf.float32), axis=2)
      # 1D: batch_size, 2D: max_num_spans, 3D: num_instances
      gold_label_mask = tf.cast(
        tf.equal(self.neighbor_tags, tags), dtype=tf.float32)
      # 1D: batch_size, 2D: max_num_spans, 3D: num_instances
      proba = self.neighbor_proba * gold_label_mask
      # 1D: batch_size, 2D: max_num_spans
      self.gold_label_proba = tf.reduce_sum(
        tf.clip_by_value(proba, 1e-10, 1.0), axis=2)

  def _build_knn_loss_op(self):
    with tf.name_scope("loss"):
      # 1D: batch_size, 2D: max_num_spans
      self.losses = tf.math.log(self.gold_label_proba)
      self.loss = - tf.reduce_mean(tf.reduce_sum(self.losses, axis=-1))
      tf.summary.scalar("loss", self.loss)

  def _build_one_nn_predict_op(self):
    with tf.name_scope("prediction"):
      neighbor_indices = tf.argmax(self.similarity, axis=2)
      knn_predicts = tf.gather(self.neighbor_tags, neighbor_indices)
      self.predicts = tf.reshape(knn_predicts,
                                 shape=(tf.shape(self.words)[0], -1))

  def _build_max_marginal_predict_op(self):
    with tf.name_scope("prediction"):
      # 1D: 1, 2D: 1, 3D: num_instances, 4D: num_tags
      one_hot_tags = tf.reshape(self.neighbor_tag_one_hots,
                                shape=[1, 1, -1, self.tag_vocab_size])
      # 1D: batch_size, 2D: max_num_spans, 3D: num_instances, 4D: 1
      proba = tf.expand_dims(self.neighbor_proba, axis=3)
      # 1D: batch_size, 2D: max_num_spans, 3D: num_instances, 4D: num_tags
      proba = proba * one_hot_tags
      # 1D: batch_size, 2D: max_num_spans, 3D: num_tags
      self.marginal_proba = tf.reduce_sum(proba, axis=2)
      self.predicts = tf.argmax(self.marginal_proba, axis=2)

  def _build_model_op(self):
    self._build_rnn_op()
    self._make_span_indices()
    if self.cfg["bilstm_type"] == "minus":
      self._build_span_minus_op()
    else:
      self._build_span_add_and_minus_op()
    self._build_span_projection_op()
    self._build_neighbor_similarity_op()
    self._build_neighbor_proba_op()

  def _build_loss_op(self):
    self._build_marginal_proba_op()
    self._build_knn_loss_op()

  def _build_predict_op(self):
    if self.cfg["predict"] == "one_nn":
      self._build_one_nn_predict_op()
    else:
      self._build_max_marginal_predict_op()

  def get_neighbor_batch(self, train_sents, train_sent_ids):
    return self.batcher.batchnize_neighbor_train_sents(
      train_sents, train_sent_ids, self.max_span_len, self.max_n_spans)

  def get_neighbor_reps_and_tags(self, span_reps, batch):
    return self.batcher.batchnize_span_reps_and_tags(
      span_reps, batch["tags"], batch["masks"])

  def get_neighbor_reps_and_tag_one_hots(self, span_reps, batch):
    return self.batcher.batchnize_span_reps_and_tag_one_hots(
      span_reps, batch["tags"], batch["masks"], self.tag_vocab_size)

  def make_one_batch_for_target(self, data, sent_id, add_tags=True):
    return self.batcher.make_each_batch_for_targets(
      batch_words=[data["words"]],
      batch_chars=[data["chars"]],
      batch_ids=[sent_id],
      max_span_len=self.max_span_len,
      max_n_spans=0,
      batch_tags=[data["tags"]] if add_tags else None)

  def _add_neighbor_instances_to_batch(self, batch, train_sents,
                                       train_sent_ids, is_train):
    if train_sent_ids:
      if is_train:
        train_sent_ids = list(set(train_sent_ids) - set(batch["instance_ids"]))
      random.shuffle(train_sent_ids)
    else:
      train_sent_ids = batch["train_sent_ids"]

    neighbor_batch = self.get_neighbor_batch(train_sents,
                                             train_sent_ids[:self.cfg["k"]])
    feed_dict = self._get_feed_dict(neighbor_batch)
    span_reps = self.sess.run([self.span_rep], feed_dict)[0]

    if is_train or self.cfg["predict"] == "one_nn":
      rep_list, tag_list = self.get_neighbor_reps_and_tags(
        span_reps, neighbor_batch)
      batch["neighbor_reps"] = rep_list
      batch["neighbor_tags"] = tag_list
    else:
      rep_list, tag_list = self.get_neighbor_reps_and_tag_one_hots(
        span_reps, neighbor_batch)
      batch["neighbor_reps"] = rep_list
      batch["neighbor_tag_one_hots"] = tag_list

    return batch

  def _make_batch_and_sample_sent_ids(self, batch, valid_record, train_sents,
                                      train_sent_ids):
    if train_sent_ids:
      random.shuffle(train_sent_ids)
      sampled_train_sent_ids = train_sent_ids[:self.cfg["k"]]
    else:
      sampled_train_sent_ids = valid_record["train_sent_ids"][:self.cfg["k"]]

    train_batch = self.batcher.make_batch_from_sent_ids(train_sents,
                                                        sampled_train_sent_ids)
    feed_dict = self._get_feed_dict(train_batch)
    span_reps = self.sess.run([self.span_rep], feed_dict)[0]
    rep_list, tag_list = self.get_neighbor_reps_and_tag_one_hots(span_reps,
                                                                 train_batch)
    batch["neighbor_reps"] = rep_list
    batch["neighbor_tag_one_hots"] = tag_list

    return batch, sampled_train_sent_ids

  def train_knn_epoch(self, batches, name):
    loss_total = 0.
    num_batches = 0
    start_time = time.time()
    train_sents = load_json(self.cfg["train_set"])
    if self.cfg["knn_sampling"] == "random":
      train_sent_ids = [sent_id for sent_id in range(len(train_sents))]
    else:
      train_sent_ids = None

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      # Setup a batch
      batch = self._add_neighbor_instances_to_batch(batch,
                                                    train_sents,
                                                    train_sent_ids,
                                                    is_train=True)
      # Convert a batch to the input format
      feed_dict = self._get_feed_dict(batch,
                                      is_train=True,
                                      keep_prob=self.cfg["keep_prob"],
                                      lr=self.cfg["lr"])
      # Train a model
      _, train_loss = self.sess.run([self.train_op, self.loss],
                                    feed_dict)

      if math.isnan(train_loss):
        self.logger.info("\n\n\nNAN: Index: %d\n" % num_batches)
        exit()

      loss_total += train_loss

    avg_loss = loss_total / num_batches
    self.logger.info("-- Time: %f seconds" % (time.time() - start_time))
    self.logger.info(
      "-- Averaged loss: %f(%f/%d)" % (avg_loss, loss_total, num_batches))
    return avg_loss, loss_total

  def evaluate_knn_epoch(self, batches, name):
    correct = 0
    p_total = 0
    num_batches = 0
    start_time = time.time()
    train_sents = load_json(self.cfg["train_set"])
    if self.cfg["knn_sampling"] == "random":
      train_sent_ids = [sent_id for sent_id in range(len(train_sents))]
    else:
      train_sent_ids = None

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      # Setup a batch
      batch = self._add_neighbor_instances_to_batch(batch,
                                                    train_sents,
                                                    train_sent_ids,
                                                    is_train=False)
      # Convert a batch to the input format
      feed_dict = self._get_feed_dict(batch)
      # Classify spans
      predicted_tags = self.sess.run([self.predicts], feed_dict)[0]

      crr_i, p_total_i = count_gold_and_system_outputs(batch["tags"],
                                                       predicted_tags,
                                                       NULL_LABEL_ID)
      correct += crr_i
      p_total += p_total_i

    p, r, f = f_score(correct, p_total, self.n_gold_spans)
    self.logger.info("-- Time: %f seconds" % (time.time() - start_time))
    self.logger.info(
      "-- {} set\tF:{:>7.2%} P:{:>7.2%} ({:>5}/{:>5}) R:{:>7.2%} ({:>5}/{:>5})"
        .format(name, f, p, correct, p_total, r, correct, self.n_gold_spans))
    return f, p, r, correct, p_total, self.n_gold_spans

  def train(self):
    self.logger.info(str(self.cfg))

    config_path = os.path.join(self.cfg["checkpoint_path"], "config.json")
    write_json(config_path, self.cfg)

    batch_size = self.cfg["batch_size"]
    epochs = self.cfg["epochs"]
    train_path = self.cfg["train_set"]
    valid_path = self.cfg["valid_set"]
    self.n_gold_spans = count_gold_spans(valid_path)

    if self.cfg["knn_sampling"] == "knn":
      self.knn_ids = h5py.File(
        os.path.join(self.cfg["raw_path"], "knn_ids.hdf5"), "r")
      valid_batch_size = 1
      shuffle = False
    else:
      valid_batch_size = batch_size
      shuffle = True

    valid_set = list(
      self.batcher.batchnize_dataset(data=valid_path,
                                     data_name="valid",
                                     batch_size=valid_batch_size,
                                     shuffle=shuffle))
    best_f1 = -np.inf
    init_lr = self.cfg["lr"]

    self.log_trainable_variables()
    self.logger.info("Start training...")
    self._add_summary()

    for epoch in range(1, epochs + 1):
      self.logger.info('Epoch {}/{}:'.format(epoch, epochs))

      train_set = self.batcher.batchnize_dataset(data=train_path,
                                                 data_name="train",
                                                 batch_size=batch_size,
                                                 shuffle=True)
      _ = self.train_knn_epoch(train_set, "train")

      if self.cfg["use_lr_decay"]:  # learning rate decay
        self.cfg["lr"] = max(init_lr / (1.0 + self.cfg["lr_decay"] * epoch),
                             self.cfg["minimal_lr"])

      eval_metrics = self.evaluate_knn_epoch(valid_set, "valid")
      cur_valid_f1 = eval_metrics[0]

      if cur_valid_f1 > best_f1:
        best_f1 = cur_valid_f1
        self.save_session(epoch)
        self.logger.info(
          '-- new BEST F1 on valid set: {:>7.2%}'.format(best_f1))

    self.train_writer.close()
    self.test_writer.close()

  def eval(self, preprocessor):
    self.logger.info(str(self.cfg))

    ########################
    # Load validation data #
    ########################
    valid_data = preprocessor.load_dataset(
      self.cfg["data_path"], keep_number=True,
      lowercase=self.cfg["char_lowercase"])
    valid_data = valid_data[:self.cfg["data_size"]]
    dataset = preprocessor.build_dataset(valid_data,
                                         self.word_dict,
                                         self.char_dict,
                                         self.tag_dict)
    dataset_path = os.path.join(self.cfg["save_path"], "tmp.json")
    write_json(dataset_path, dataset)
    self.logger.info("Valid sentences: {:>7}".format(len(dataset)))
    self.n_gold_spans = count_gold_spans(dataset_path)

    ######################
    # Load training data #
    ######################
    train_sents = load_json(self.cfg["train_set"])
    if self.cfg["knn_sampling"] == "random":
      train_sent_ids = [sent_id for sent_id in range(len(train_sents))]
    else:
      train_sent_ids = None
    self.logger.info("Train sentences: {:>7}".format(len(train_sents)))

    #############
    # Main loop #
    #############
    correct = 0
    p_total = 0
    start_time = time.time()

    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]

      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      batch = self.make_one_batch_for_target(data, valid_sent_id)

      #####################
      # Sentence sampling #
      #####################
      batch, sampled_sent_ids = self._make_batch_and_sample_sent_ids(
        batch, record, train_sents, train_sent_ids)

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      batch_sims, batch_preds = self.sess.run(
        [self.similarity, self.predicts], feed_dict)

      crr_i, p_total_i = count_gold_and_system_outputs(
        batch["tags"], batch_preds, NULL_LABEL_ID)
      correct += crr_i
      p_total += p_total_i

    ##############
    # Evaluation #
    ##############
    p, r, f = f_score(correct, p_total, self.n_gold_spans)
    self.logger.info("-- Time: %f seconds" % (time.time() - start_time))
    self.logger.info(
      "-- F:{:>7.2%} P:{:>7.2%} ({:>5}/{:>5}) R:{:>7.2%} ({:>5}/{:>5})"
        .format(f, p, correct, p_total, r, correct, self.n_gold_spans))

  def save_predicted_spans(self, data_name, preprocessor):
    self.logger.info(str(self.cfg))

    ########################
    # Load validation data #
    ########################
    valid_data = preprocessor.load_dataset(
      self.cfg["data_path"], keep_number=True,
      lowercase=self.cfg["char_lowercase"])
    valid_data = valid_data[:self.cfg["data_size"]]
    dataset = preprocessor.build_dataset(valid_data,
                                         self.word_dict,
                                         self.char_dict,
                                         self.tag_dict)
    dataset_path = os.path.join(self.cfg["save_path"], "tmp.json")
    write_json(dataset_path, dataset)
    self.logger.info("Valid sentences: {:>7}".format(len(dataset)))

    ######################
    # Load training data #
    ######################
    train_sents = load_json(self.cfg["train_set"])
    if self.cfg["knn_sampling"] == "random":
      train_sent_ids = [sent_id for sent_id in range(len(train_sents))]
    else:
      train_sent_ids = None
    self.logger.info("Train sentences: {:>7}".format(len(train_sents)))

    #############
    # Main loop #
    #############
    start_time = time.time()
    results = []
    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]
      batch = self.make_one_batch_for_target(data, valid_sent_id,
                                             add_tags=False)
      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      #####################
      # Sentence sampling #
      #####################
      batch, sampled_sent_ids = self._make_batch_and_sample_sent_ids(
        batch, record, train_sents, train_sent_ids)

      ###############
      # KNN predict #
      ###############
      feed_dict = self._get_feed_dict(batch)
      batch_preds = self.sess.run([self.predicts], feed_dict)[0]
      preds = batch_preds[0]

      ########################
      # Make predicted spans #
      ########################
      indx_i, indx_j = get_span_indices(n_words=len(record["words"]),
                                        max_span_len=self.max_span_len)
      assert len(preds) == len(indx_i) == len(indx_j)
      pred_spans = [[self.rev_tag_dict[pred_label_id], int(i), int(j)]
                    for pred_label_id, i, j in zip(preds, indx_i, indx_j)
                    if pred_label_id != NULL_LABEL_ID]

      ##################
      # Add the result #
      ##################
      results.append({"sent_id": valid_sent_id,
                      "words": record["words"],
                      "spans": pred_spans,
                      "train_sent_ids": sampled_sent_ids})

    path = os.path.join(self.cfg["checkpoint_path"],
                        "%s.predicted_spans.json" % data_name)
    write_json(path, results)
    self.logger.info(
      "-- Time: %f seconds\nFINISHED." % (time.time() - start_time))

  def save_predicted_bio_tags(self, data_name, preprocessor):
    self.logger.info(str(self.cfg))

    ########################
    # Load validation data #
    ########################
    valid_data = preprocessor.load_dataset(
      self.cfg["data_path"], keep_number=True,
      lowercase=self.cfg["char_lowercase"])
    valid_data = valid_data[:self.cfg["data_size"]]
    dataset = preprocessor.build_dataset(valid_data,
                                         self.word_dict,
                                         self.char_dict,
                                         self.tag_dict)
    dataset_path = os.path.join(self.cfg["save_path"], "tmp.json")
    write_json(dataset_path, dataset)
    self.logger.info("Valid sentences: {:>7}".format(len(dataset)))

    ######################
    # Load training data #
    ######################
    train_sents = load_json(self.cfg["train_set"])
    if self.cfg["knn_sampling"] == "random":
      train_sent_ids = [sent_id for sent_id in range(len(train_sents))]
    else:
      train_sent_ids = None
    self.logger.info("Train sentences: {:>7}".format(len(train_sents)))

    #############
    # Main loop #
    #############
    start_time = time.time()
    path = os.path.join(self.cfg["checkpoint_path"], "%s.bio.txt" % data_name)
    fout_txt = open(path, "w")
    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]
      batch = self.make_one_batch_for_target(data, valid_sent_id,
                                             add_tags=False)
      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      #####################
      # Sentence sampling #
      #####################
      batch, sampled_sent_ids = self._make_batch_and_sample_sent_ids(
        batch, record, train_sents, train_sent_ids)

      ###############
      # KNN predict #
      ###############
      feed_dict = self._get_feed_dict(batch)
      proba = self.sess.run([self.marginal_proba], feed_dict)[0][0]

      ########################
      # Make predicted spans #
      ########################
      words = record["words"]
      triples = greedy_search(proba,
                              n_words=len(words),
                              max_span_len=self.max_span_len,
                              null_label_id=NULL_LABEL_ID)
      pred_bio_tags = span2bio(spans=triples,
                               n_words=len(words),
                               tag_dict=self.rev_tag_dict)
      gold_bio_tags = span2bio(spans=record["tags"],
                               n_words=len(words))
      assert len(words) == len(pred_bio_tags) == len(gold_bio_tags)

      ####################
      # Write the result #
      ####################
      for word, gold_tag, pred_tag in zip(words, gold_bio_tags, pred_bio_tags):
        fout_txt.write("%s _ %s %s\n" % (word, gold_tag, pred_tag))
      fout_txt.write("\n")

    self.logger.info(
      "-- Time: %f seconds\nFINISHED." % (time.time() - start_time))

  def save_nearest_spans(self, data_name, preprocessor, print_knn):
    self.logger.info(str(self.cfg))

    ########################
    # Load validation data #
    ########################
    valid_data = preprocessor.load_dataset(
      self.cfg["data_path"], keep_number=True,
      lowercase=self.cfg["char_lowercase"])
    valid_data = valid_data[:self.cfg["data_size"]]
    dataset = preprocessor.build_dataset(valid_data,
                                         self.word_dict,
                                         self.char_dict,
                                         self.tag_dict)
    dataset_path = os.path.join(self.cfg["save_path"], "tmp.json")
    write_json(dataset_path, dataset)
    self.logger.info("Valid sentences: {:>7}".format(len(dataset)))
    self.n_gold_spans = count_gold_spans(dataset_path)

    ######################
    # Load training data #
    ######################
    train_sents = load_json(self.cfg["train_set"])
    if self.cfg["knn_sampling"] == "random":
      train_sent_ids = [sent_id for sent_id in range(len(train_sents))]
    else:
      train_sent_ids = None
    train_data = preprocessor.load_dataset(
      os.path.join(self.cfg["raw_path"], "train.json"),
      keep_number=True, lowercase=False)
    self.logger.info("Train sentences: {:>7}".format(len(train_sents)))

    #############
    # Main loop #
    #############
    correct = 0
    p_total = 0
    start_time = time.time()
    file_path = os.path.join(self.cfg["checkpoint_path"],
                             "%s.nearest_spans.txt" % data_name)
    fout_txt = open(file_path, "w")
    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]
      batch = self.make_one_batch_for_target(data, valid_sent_id)

      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      #####################
      # Sentence sampling #
      #####################
      batch, sampled_sent_ids = self._make_batch_and_sample_sent_ids(
        batch, record, train_sents, train_sent_ids)

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      batch_sims, batch_preds = self.sess.run(
        [self.similarity, self.predicts], feed_dict)

      crr_i, p_total_i = count_gold_and_system_outputs(
        batch["tags"], batch_preds, NULL_LABEL_ID)
      correct += crr_i
      p_total += p_total_i

      ####################
      # Write the result #
      ####################
      self._write_predictions(fout_txt, record)
      self._write_nearest_spans(
        fout_txt, record, train_data, sampled_sent_ids, batch_sims,
        batch_preds, print_knn)

    fout_txt.close()

    p, r, f = f_score(correct, p_total, self.n_gold_spans)
    self.logger.info("-- Time: %f seconds" % (time.time() - start_time))
    self.logger.info(
      "-- {} set\tF:{:>7.2%} P:{:>7.2%} ({:>5}/{:>5}) R:{:>7.2%} ({:>5}/{:>5})"
        .format(data_name, f, p, correct, p_total, r, correct,
                self.n_gold_spans))

  @staticmethod
  def _write_predictions(fout_txt, record):
    fout_txt.write("-SENT:%d || %s || %s\n" % (
      record["sent_id"],
      " ".join(record["words"]),
      " ".join(["(%s,%d,%d)" % (r, i, j) for (r, i, j) in record["tags"]])))

  def _write_nearest_spans(self, fout_txt, record, train_data,
                           sampled_sent_ids, batch_sims, batch_preds,
                           print_knn):

    def _write_train_sents(_sampled_train_sents):
      for _train_record in _sampled_train_sents:
        fout_txt.write("--kNN:%d || %s || %s\n" % (
          _train_record["sent_id"],
          " ".join(_train_record["words"]),
          " ".join(["(%s,%d,%d)" % (r, i, j)
                    for (r, i, j) in _train_record["tags"]])))

    def _write_gold_and_pred_spans(_record, _pred_label_id, _span_boundaries):
      if (i, j) in _span_boundaries:
        _index = _span_boundaries.index((i, j))
        gold_label = _record["tags"][_index][0]
      else:
        gold_label = "O"

      pred_label = self.rev_tag_dict[_pred_label_id]
      fout_txt.write("##(%d,%d) || %s || %s || %s\n" % (
        i, j, " ".join(record["words"][i: j + 1]), pred_label, gold_label))

    def _get_nearest_spans(_sampled_train_sents):
      _nearest_spans = []
      _prev_indx = 0
      _temp_indx = 0
      for _record in _sampled_train_sents:
        _indx_i, _indx_j = get_span_indices(n_words=len(_record["words"]),
                                            max_span_len=self.max_span_len)
        _temp_indx += len(_indx_i)
        _temp_scores = scores[_prev_indx: _temp_indx]
        assert len(_temp_scores) == len(_indx_i) == len(_indx_j)
        _nearest_spans.extend(
          get_scores_and_spans(spans=_record["tags"],
                               scores=_temp_scores,
                               sent_id=_record["sent_id"],
                               indx_i=_indx_i,
                               indx_j=_indx_j))
        _prev_indx = _temp_indx
      return _nearest_spans

    def _write_nearest_spans_for_each_span(_sampled_train_sents):
      nearest_spans = _get_nearest_spans(_sampled_train_sents)
      nearest_spans.sort(key=lambda span: span[-1], reverse=True)
      for rank, (r, sent_id, i, j, score) in enumerate(nearest_spans[:10]):
        mention = " ".join(train_data[sent_id]["words"][i: j + 1])
        text = "{} || {} || sent:{} || ({},{}) || {:.3g}".format(
          r, mention, sent_id, i, j, score)
        fout_txt.write("####RANK:%d %s\n" % (rank, text))

    sampled_train_sents = [train_data[sent_id]
                           for sent_id in sampled_sent_ids]
    if print_knn:
      _write_train_sents(sampled_train_sents)

    sims = batch_sims[0]  # 1D: n_spans, 2D: n_instances
    preds = batch_preds[0]  # 1D: n_spans
    indx_i, indx_j = get_span_indices(n_words=len(record["words"]),
                                      max_span_len=self.max_span_len)
    span_boundaries = [(i, j) for _, i, j in record["tags"]]

    assert len(sims) == len(preds) == len(indx_i) == len(indx_j)
    for scores, pred_label_id, i, j in zip(sims, preds, indx_i, indx_j):
      if pred_label_id == NULL_LABEL_ID and (i, j) not in span_boundaries:
        continue
      _write_gold_and_pred_spans(record, pred_label_id, span_boundaries)
      _write_nearest_spans_for_each_span(sampled_train_sents)

    fout_txt.write("\n")

  def save_span_representation(self, data_name, preprocessor):
    self.logger.info(str(self.cfg))

    ########################
    # Load validation data #
    ########################
    valid_data = preprocessor.load_dataset(
      self.cfg["data_path"], keep_number=True,
      lowercase=self.cfg["char_lowercase"])
    valid_data = valid_data[:self.cfg["data_size"]]
    dataset = preprocessor.build_dataset(valid_data,
                                         self.word_dict,
                                         self.char_dict,
                                         self.tag_dict)
    self.logger.info("Valid sentences: {:>7}".format(len(dataset)))

    #############
    # Main loop #
    #############
    start_time = time.time()
    gold_labels = {}
    fout_path = os.path.join(self.cfg["checkpoint_path"],
                             "%s.span_reps.hdf5" % data_name)
    fout = h5py.File(fout_path, 'w')

    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]
      batch = self.make_one_batch_for_target(data, valid_sent_id)

      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      span_reps = self.sess.run([self.span_rep], feed_dict)[0][0]
      span_tags = batch["tags"][0]
      assert len(span_reps) == len(span_tags)

      ##################
      # Add the result #
      ##################
      fout.create_dataset(
        name='{}'.format(valid_sent_id),
        dtype='float32',
        data=span_reps)
      gold_labels[valid_sent_id] = [self.rev_tag_dict[int(tag)]
                                    for tag in span_tags]
    fout.close()
    path = os.path.join(self.cfg["checkpoint_path"],
                        "%s.gold_labels.json" % data_name)
    write_json(path, gold_labels)
    self.logger.info(
      "-- Time: %f seconds\nFINISHED." % (time.time() - start_time))

  def predict_on_command_line(self, preprocessor):

    def _load_glove(glove_path):
      vocab = {}
      vectors = []
      total = int(4e5)
      with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Load glove"):
          line = line.lstrip().rstrip().split(" ")
          vocab[line[0]] = len(vocab)
          vectors.append([float(x) for x in line[1:]])
      assert len(vocab) == len(vectors)
      return vocab, np.asarray(vectors)

    def _mean_vectors(sents, emb, vocab):
      unk_vec = np.zeros(emb.shape[1])
      mean_vecs = []
      for words in sents:
        vecs = []
        for word in words:
          word = word.lower()
          if word in vocab:
            vec = emb[vocab[word]]
          else:
            vec = unk_vec
          vecs.append(vec)
        mean_vecs.append(np.mean(vecs, axis=0))
      return mean_vecs

    def _cosine_sim(p0, p1):
      d = (norm(p0) * norm(p1))
      if d > 0:
        return np.dot(p0, p1) / d
      return 0.0

    def _setup_repository(_train_sents, _train_data=None):
      if self.cfg["knn_sampling"] == "random":
        _train_sent_ids = [_sent_id for _sent_id in range(len(_train_sents))]
        _vocab = _glove = _train_embs = None
      else:
        _train_sent_ids = None
        _vocab, _glove = _load_glove("data/emb/glove.6B.100d.txt")
        _train_words = [[w.lower() for w in _train_record["words"]]
                        for _train_record in _train_data]
        _train_embs = _mean_vectors(_train_words, _glove, _vocab)
      return _train_sent_ids, _train_embs, _vocab, _glove

    def _make_ids(_words):
      _char_ids = []
      _word_ids = []
      for word in _words:
        _char_ids.append([self.char_dict[char]
                          if char in self.char_dict else self.char_dict[UNK]
                          for char in word])
        word = word_convert(word, keep_number=False, lowercase=True)
        _word_ids.append(self.word_dict[word]
                         if word in self.word_dict else self.word_dict[UNK])
      return _char_ids, _word_ids

    def _retrieve_knn_train_sents(_record, _train_embs, _vocab, _glove):
      test_words = [w.lower() for w in _record["words"]]
      test_emb = _mean_vectors([test_words], _glove, _vocab)[0]
      sim = [_cosine_sim(train_emb, test_emb) for train_emb in _train_embs]
      arg_sort = np.argsort(sim)[::-1][:self.cfg["k"]]
      _record["train_sent_ids"] = [int(arg) for arg in arg_sort]
      return _record

    def _get_nearest_spans(_sampled_train_sents, _scores):
      _nearest_spans = []
      _prev_indx = 0
      _temp_indx = 0
      for _record in _sampled_train_sents:
        _indx_i, _indx_j = get_span_indices(n_words=len(_record["words"]),
                                            max_span_len=self.max_span_len)
        _temp_indx += len(_indx_i)
        _temp_scores = _scores[_prev_indx: _temp_indx]
        assert len(_temp_scores) == len(_indx_i) == len(_indx_j)
        _nearest_spans.extend(
          get_scores_and_spans(spans=_record["tags"],
                               scores=_temp_scores,
                               sent_id=_record["sent_id"],
                               indx_i=_indx_i,
                               indx_j=_indx_j))
        _prev_indx = _temp_indx
      _nearest_spans.sort(key=lambda span: span[-1], reverse=True)
      return _nearest_spans

    ######################
    # Load training data #
    ######################
    train_sents = load_json(self.cfg["train_set"])
    train_data = preprocessor.load_dataset(
      os.path.join(self.cfg["raw_path"], "train.json"),
      keep_number=True, lowercase=False)
    train_sent_ids, train_embs, vocab, glove = _setup_repository(
      train_sents, train_data)

    ########################################
    # Load each sentence from command line #
    ########################################
    print("\nPREDICTION START\n")
    while True:
      sentence = input('\nEnter a tokenized sentence: ')
      words = sentence.split()
      char_ids, word_ids = _make_ids(words)
      data = {"words": word_ids, "chars": char_ids}
      record = {"sent_id": 0, "words": words, "train_sent_ids": None}
      batch = self.make_one_batch_for_target(data, sent_id=0, add_tags=False)

      #####################
      # Sentence sampling #
      #####################
      if self.cfg["knn_sampling"] == "knn":
        record = _retrieve_knn_train_sents(record, train_embs, vocab, glove)
      batch, sampled_sent_ids = self._make_batch_and_sample_sent_ids(
        batch, record, train_sents, train_sent_ids)

      ##############
      # Prediction #
      ##############
      feed_dict = self._get_feed_dict(batch)
      batch_sims, batch_preds = self.sess.run(
        [self.similarity, self.predicts], feed_dict)

      ####################
      # Write the result #
      ####################
      sims = batch_sims[0]  # 1D: n_spans, 2D: n_instances
      preds = batch_preds[0]  # 1D: n_spans
      indx_i, indx_j = get_span_indices(n_words=len(record["words"]),
                                        max_span_len=self.max_span_len)

      assert len(sims) == len(preds) == len(indx_i) == len(indx_j)
      sampled_train_sents = [train_data[sent_id]
                             for sent_id in sampled_sent_ids]

      for scores, pred_label_id, i, j in zip(sims, preds, indx_i, indx_j):
        if pred_label_id == NULL_LABEL_ID:
          continue
        pred_label = self.rev_tag_dict[pred_label_id]
        print("#(%d,%d) || %s || %s" % (
          i, j, " ".join(record["words"][i: j + 1]), pred_label))

        nearest_spans = _get_nearest_spans(sampled_train_sents, scores)
        for k, (r, _sent_id, a, b, _score) in enumerate(nearest_spans[:5]):
          train_words = train_data[_sent_id]["words"]
          if a - 5 < 0:
            left_context = ""
          else:
            left_context = " ".join(train_words[a - 5: a])
            left_context = "... " + left_context
          right_context = " ".join(train_words[b + 1: b + 6])
          if b + 6 < len(train_words):
            right_context = right_context + " ..."
          mention = " ".join(train_words[a: b + 1])
          text = "{}: {} [{}] {}".format(
            r, left_context, mention, right_context)
          print("## %d %s" % (k, text))
