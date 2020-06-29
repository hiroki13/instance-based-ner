# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time

import h5py
import numpy as np

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import \
  stack_bidirectional_dynamic_rnn

from models import BaseModel
from models.decoders import get_span_indices, greedy_search
from models.network_components import multi_conv1d, highway_network
from utils.common import write_json
from utils.data_utils import metrics_for_multi_class_spans, f_score, \
  count_gold_spans, count_gold_and_system_outputs, span2bio

NULL_LABEL_ID = 0


class SpanModel(BaseModel):

  def __init__(self, config, batcher, is_train=True):
    self.max_span_len = config["max_span_len"]
    self.n_gold_spans = None
    self.proba = None
    super(SpanModel, self).__init__(config, batcher, is_train)

  def _add_placeholders(self):
    self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
    self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")
    self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
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
    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr
    return feed_dict

  def _create_rnn_cell(self):
    if self.cfg["num_layers"] is None or self.cfg["num_layers"] <= 1:
      return self._create_single_rnn_cell(self.cfg["num_units"])
    else:
      if self.cfg["use_stack_rnn"]:
        lstm_cells = []
        for i in range(self.cfg["num_layers"]):
          cell = tf.nn.rnn_cell.LSTMCell(self.cfg["num_units"],
                                         initializer=tf.initializers.orthogonal
                                         )
          cell = tf.contrib.rnn.DropoutWrapper(cell,
                                               state_keep_prob=self.keep_prob,
                                               input_keep_prob=self.keep_prob,
                                               dtype=tf.float32)
          lstm_cells.append(cell)
        return lstm_cells
      else:
        return MultiRNNCell(
          [self._create_single_rnn_cell(self.cfg["num_units"])
           for _ in range(self.cfg["num_layers"])])

  def _build_embedding_op(self):
    with tf.variable_scope("embeddings"):
      if not self.cfg["use_pretrained"]:
        self.word_embeddings = tf.get_variable(name="emb",
                                               dtype=tf.float32,
                                               trainable=True,
                                               shape=[self.word_vocab_size,
                                                      self.cfg["emb_dim"]])
      else:
        padding_token_emb = tf.get_variable(name="padding_emb",
                                            dtype=tf.float32,
                                            trainable=False,
                                            shape=[1, self.cfg["emb_dim"]])
        special_token_emb = tf.get_variable(name="spacial_emb",
                                            dtype=tf.float32,
                                            trainable=True,
                                            shape=[2, self.cfg["emb_dim"]])
        token_emb = tf.Variable(
          np.load(self.cfg["pretrained_emb"])["embeddings"],
          name="emb", dtype=tf.float32, trainable=self.cfg["tuning_emb"])
        self.word_embeddings = tf.concat(
          [padding_token_emb, special_token_emb, token_emb], axis=0)

      word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words,
                                        name="words_emb")
      print("word embedding shape: {}".format(word_emb.get_shape().as_list()))

      if self.cfg["use_chars"]:
        self.char_embeddings = tf.get_variable(name="char_emb",
                                               dtype=tf.float32,
                                               trainable=True,
                                               shape=[self.char_vocab_size,
                                                      self.cfg["char_emb_dim"]]
                                               )
        char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.chars,
                                          name="chars_emb")
        char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"],
                                      self.cfg["channel_sizes"],
                                      drop_rate=self.drop_rate,
                                      is_train=self.is_train)
        print("chars representation shape: {}".format(
          char_represent.get_shape().as_list()))
        word_emb = tf.concat([word_emb, char_represent], axis=-1)

      if self.cfg["use_highway"]:
        self.word_emb = highway_network(word_emb, self.cfg["highway_layers"],
                                        use_bias=True, bias_init=0.0,
                                        keep_prob=self.keep_prob,
                                        is_train=self.is_train)
      else:
        self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate,
                                          training=self.is_train)
      print("word and chars concatenation shape: {}".format(
        self.word_emb.get_shape().as_list()))

  def _build_rnn_op(self):
    with tf.variable_scope("bi_directional_rnn"):
      cell_fw = self._create_rnn_cell()
      cell_bw = self._create_rnn_cell()

      if self.cfg["use_stack_rnn"]:
        rnn_outs, *_ = stack_bidirectional_dynamic_rnn(
          cell_fw, cell_bw, self.word_emb, dtype=tf.float32)
      else:
        rnn_outs, *_ = bidirectional_dynamic_rnn(
          cell_fw, cell_bw, self.word_emb, dtype=tf.float32)
      rnn_outs = tf.concat(rnn_outs, axis=-1)
      rnn_outs = tf.layers.dropout(rnn_outs, rate=self.drop_rate,
                                   training=self.is_train)
      self.rnn_outs = rnn_outs
      print("rnn output shape: {}".format(rnn_outs.get_shape().as_list()))

  def _make_span_indices(self):
    with tf.name_scope("span_indices"):
      n_words = tf.shape(self.rnn_outs)[1]
      n_spans = tf.cast(n_words * (n_words + 1) / 2, dtype=tf.int32)
      ones = tf.contrib.distributions.fill_triangular(tf.ones(shape=[n_spans]),
                                                      upper=True)
      num_upper = tf.minimum(n_words, self.cfg["max_span_len"] - 1)
      ones = tf.linalg.band_part(ones, num_lower=tf.cast(0, dtype=tf.int32),
                                 num_upper=num_upper)
      self.span_indices = tf.transpose(
        tf.where(tf.not_equal(ones, tf.constant(0, dtype=tf.float32))))

  def _build_span_minus_op(self):
    with tf.variable_scope("rnn_span_rep"):
      i = self.span_indices[0]
      j = self.span_indices[1]
      batch_size = tf.shape(self.rnn_outs)[0]
      dim = self.cfg["num_units"]
      x_fw = self.rnn_outs[:, :, :dim]
      x_bw = self.rnn_outs[:, :, dim:]

      pad = tf.zeros(shape=(batch_size, 1, dim), dtype=tf.float32)
      x_fw_pad = tf.concat([pad, x_fw], axis=1)
      x_bw_pad = tf.concat([x_bw, pad], axis=1)

      h_fw_i = tf.gather(x_fw_pad, i, axis=1)
      h_fw_j = tf.gather(x_fw, j, axis=1)
      h_bw_i = tf.gather(x_bw, i, axis=1)
      h_bw_j = tf.gather(x_bw_pad, j + 1, axis=1)

      span_fw = h_fw_j - h_fw_i
      span_bw = h_bw_i - h_bw_j
      self.rnn_span_rep = tf.concat([span_fw, span_bw], axis=-1)

      print("rnn span rep shape: {}".format(
        self.rnn_span_rep.get_shape().as_list()))

  def _build_span_add_and_minus_op(self):
    with tf.variable_scope("rnn_span_rep"):
      i = self.span_indices[0]
      j = self.span_indices[1]
      batch_size = tf.shape(self.rnn_outs)[0]
      dim = self.cfg["num_units"]
      x_fw = self.rnn_outs[:, :, :dim]
      x_bw = self.rnn_outs[:, :, dim:]

      pad = tf.zeros(shape=(batch_size, 1, dim), dtype=tf.float32)
      x_fw_pad = tf.concat([pad, x_fw], axis=1)
      x_bw_pad = tf.concat([x_bw, pad], axis=1)

      h_fw_i = tf.gather(x_fw, i, axis=1)
      h_fw_i_pad = tf.gather(x_fw_pad, i, axis=1)
      h_fw_j = tf.gather(x_fw, j, axis=1)
      h_bw_i = tf.gather(x_bw, i, axis=1)
      h_bw_j = tf.gather(x_bw, j, axis=1)
      h_bw_j_pad = tf.gather(x_bw_pad, j + 1, axis=1)

      span_add_fw = h_fw_i + h_fw_j
      span_add_bw = h_bw_i + h_bw_j
      span_minus_fw = h_fw_j - h_fw_i_pad
      span_minus_bw = h_bw_i - h_bw_j_pad
      self.rnn_span_rep = tf.concat(
        [span_add_fw, span_add_bw, span_minus_fw, span_minus_bw], axis=-1)

      print("rnn span rep shape: {}".format(
        self.rnn_span_rep.get_shape().as_list()))

  def _build_span_projection_op(self):
    with tf.variable_scope("span_projection"):
      span_rep = tf.layers.dense(self.rnn_span_rep,
                                 units=self.cfg["num_units"],
                                 use_bias=True)
      self.span_rep = tf.layers.dropout(span_rep,
                                        rate=self.drop_rate,
                                        training=self.is_train)
    print("span rep shape: {}".format(self.span_rep.get_shape().as_list()))

  def _build_label_projection_with_null_zero_op(self):
    with tf.variable_scope("label_projection"):
      null_label_emb = tf.get_variable(name="null_label_emb",
                                       trainable=False,
                                       shape=[1, self.cfg["num_units"]])
      label_emb = tf.get_variable(name="label_emb",
                                  dtype=tf.float32,
                                  trainable=True,
                                  shape=[self.tag_vocab_size - 1,
                                         self.cfg["num_units"]])
      self.label_embeddings = tf.concat([null_label_emb, label_emb], axis=0)
      self.logits = tf.tensordot(self.span_rep, self.label_embeddings,
                                 axes=[-1, -1])
      print("logits shape: {}".format(self.logits.get_shape().as_list()))

  def _build_model_op(self):
    self._build_rnn_op()
    self._make_span_indices()

    if self.cfg["bilstm_type"] == "minus":
      self._build_span_minus_op()
    else:
      self._build_span_add_and_minus_op()

    self._build_span_projection_op()
    self._build_label_projection_with_null_zero_op()

  def _build_loss_op(self):
    self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.logits, labels=self.tags)
    self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=-1))
    tf.summary.scalar("loss", self.loss)

  def _build_train_op(self):
    with tf.variable_scope("train_step"):
      optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
      if self.cfg["grad_clip"] is not None and self.cfg["grad_clip"] > 0:
        grads, vs = zip(*optimizer.compute_gradients(self.loss))
        grads, _ = tf.clip_by_global_norm(grads, self.cfg["grad_clip"])
        self.train_op = optimizer.apply_gradients(zip(grads, vs))
      else:
        self.train_op = optimizer.minimize(self.loss)

  def _build_predict_op(self):
    self.predicts = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

  def build_proba_op(self):
    self.proba = tf.nn.softmax(self.logits)

  def train_epoch(self, batches):
    loss_total = 0.
    correct = 0
    p_total = 0
    r_total = 0
    num_batches = 0
    start_time = time.time()

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      feed_dict = self._get_feed_dict(batch, is_train=True,
                                      keep_prob=self.cfg["keep_prob"],
                                      lr=self.cfg["lr"])
      outputs = self.sess.run([self.train_op, self.loss, self.predicts],
                              feed_dict)
      _, train_loss, predicts = outputs

      loss_total += train_loss
      crr_i, p_total_i, r_total_i = metrics_for_multi_class_spans(
        batch["tags"], predicts, NULL_LABEL_ID)
      correct += crr_i
      p_total += p_total_i
      r_total += r_total_i

    avg_loss = loss_total / num_batches
    p, r, f = f_score(correct, p_total, r_total)

    self.logger.info("-- Time: %f seconds" % (time.time() - start_time))
    self.logger.info(
      "-- Averaged loss: %f(%f/%d)" % (avg_loss, loss_total, num_batches))
    self.logger.info(
      "-- {} set\tF:{:>7.2%} P:{:>7.2%} ({:>5}/{:>5}) R:{:>7.2%} ({:>5}/{:>5})"
        .format("train", f, p, correct, p_total, r, correct, r_total))
    return avg_loss, loss_total

  def evaluate_epoch(self, batches, name):
    correct = 0
    p_total = 0
    num_batches = 0
    start_time = time.time()

    for batch in batches:
      num_batches += 1
      if num_batches % 100 == 0:
        print("%d" % num_batches, flush=True, end=" ")

      feed_dict = self._get_feed_dict(batch)
      predicts = self.sess.run(self.predicts, feed_dict)
      crr_i, p_total_i = count_gold_and_system_outputs(
        batch["tags"], predicts, NULL_LABEL_ID)
      correct += crr_i
      p_total += p_total_i

    p, r, f = f_score(correct, p_total, self.n_gold_spans)
    self.logger.info('-- Time: %f seconds' % (time.time() - start_time))
    self.logger.info(
      "-- {} set\tF:{:>7.2%} P:{:>7.2%} ({:>5}/{:>5}) R:{:>7.2%} ({:>5}/{:>5})"
        .format(name, f, p, correct, p_total, r, correct, self.n_gold_spans))
    return f, p, r, correct, p_total, self.n_gold_spans

  def train(self):
    self.logger.info(str(self.cfg))
    write_json(os.path.join(self.cfg["checkpoint_path"], "config.json"),
               self.cfg)

    batch_size = self.cfg["batch_size"]
    epochs = self.cfg["epochs"]
    train_path = self.cfg["train_set"]
    valid_path = self.cfg["valid_set"]
    self.n_gold_spans = count_gold_spans(valid_path)
    valid_set = list(
      self.batcher.batchnize_dataset(valid_path, batch_size, shuffle=True))

    best_f1 = -np.inf
    init_lr = self.cfg["lr"]

    self.log_trainable_variables()
    self.logger.info("Start training...")
    self._add_summary()
    for epoch in range(1, epochs + 1):
      self.logger.info('Epoch {}/{}:'.format(epoch, epochs))

      train_set = self.batcher.batchnize_dataset(train_path, batch_size,
                                                 shuffle=True)
      _ = self.train_epoch(train_set)

      if self.cfg["use_lr_decay"]:  # learning rate decay
        self.cfg["lr"] = max(init_lr / (1.0 + self.cfg["lr_decay"] * epoch),
                             self.cfg["minimal_lr"])

      eval_metrics = self.evaluate_epoch(valid_set, "valid")
      cur_valid_f1 = eval_metrics[0]

      if cur_valid_f1 > best_f1:
        best_f1 = cur_valid_f1
        self.save_session(epoch)
        self.logger.info(
          "-- new BEST F1 on valid set: {:>7.2%}".format(best_f1))

    self.train_writer.close()
    self.test_writer.close()

  def eval(self, preprocessor):
    self.logger.info(str(self.cfg))
    data = preprocessor.load_dataset(self.cfg["data_path"],
                                     keep_number=True,
                                     lowercase=self.cfg["char_lowercase"])
    data = data[:self.cfg["data_size"]]
    dataset = preprocessor.build_dataset(data, self.word_dict,
                                         self.char_dict, self.tag_dict)
    write_json(os.path.join(self.cfg["save_path"], "tmp.json"), dataset)
    self.n_gold_spans = count_gold_spans(
      os.path.join(self.cfg["save_path"], "tmp.json"))
    self.logger.info("Target data: %s sentences" % len(dataset))
    del dataset

    batches = list(self.batcher.batchnize_dataset(
      os.path.join(self.cfg["save_path"], "tmp.json"),
      batch_size=self.cfg["batch_size"], shuffle=True))
    self.logger.info("Target data: %s batches" % len(batches))
    _ = self.evaluate_epoch(batches, "valid")

  def make_one_batch(self, data, add_tags=True):
    return self.batcher.make_each_batch(
      batch_words=[data["words"]],
      batch_chars=[data["chars"]],
      max_span_len=self.max_span_len,
      batch_tags=[data["tags"]] if add_tags else None)

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

    #############
    # Main loop #
    #############
    start_time = time.time()
    results = []
    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]
      batch = self.batcher.make_each_batch(
        batch_words=[data["words"]], batch_chars=[data["chars"]],
        max_span_len=self.max_span_len)

      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      #################
      # Predict spans #
      #################
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
                      "spans": pred_spans})

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

    #############
    # Main loop #
    #############
    start_time = time.time()
    path = os.path.join(self.cfg["checkpoint_path"], "%s.bio.txt" % data_name)
    fout_txt = open(path, "w")
    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]
      batch = self.make_one_batch(data, add_tags=False)

      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      #################
      # Predict spans #
      #################
      feed_dict = self._get_feed_dict(batch)
      proba = self.sess.run([self.proba], feed_dict)[0][0]

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

  def save_span_representation(self, data_name, preprocessor):
    self.logger.info(str(self.cfg))

    ########################
    # Load validation data #
    ########################
    valid_data = preprocessor.load_dataset(
      self.cfg["data_path"], keep_number=True,
      lowercase=self.cfg["char_lowercase"])
    valid_data = valid_data[:self.cfg["data_size"]]
    dataset = preprocessor.build_dataset(valid_data, self.word_dict,
                                         self.char_dict, self.tag_dict)
    dataset_path = os.path.join(self.cfg["save_path"], "tmp.json")
    write_json(dataset_path, dataset)
    self.logger.info("Valid sentences: {:>7}".format(len(dataset)))

    #############
    # Main loop #
    #############
    start_time = time.time()
    results = []
    fout_hdf5 = h5py.File(os.path.join(self.cfg["checkpoint_path"],
                                       "%s.span_reps.hdf5" % data_name), 'w')
    print("PREDICTION START")
    for record, data in zip(valid_data, dataset):
      valid_sent_id = record["sent_id"]
      batch = self.batcher.make_each_batch(
        batch_words=[data["words"]], batch_chars=[data["chars"]],
        max_span_len=self.max_span_len, batch_tags=[data["tags"]])

      if (valid_sent_id + 1) % 100 == 0:
        print("%d" % (valid_sent_id + 1), flush=True, end=" ")

      #################
      # Predict spans #
      #################
      feed_dict = self._get_feed_dict(batch)
      preds, span_reps = self.sess.run([self.predicts, self.span_rep],
                                       feed_dict=feed_dict)
      golds = batch["tags"][0]
      preds = preds[0]
      span_reps = span_reps[0]
      assert len(span_reps) == len(golds) == len(preds)

      ########################
      # Make predicted spans #
      ########################
      indx_i, indx_j = get_span_indices(n_words=len(record["words"]),
                                        max_span_len=self.max_span_len)
      assert len(preds) == len(indx_i) == len(indx_j)
      pred_spans = [[self.rev_tag_dict[label_id], int(i), int(j)]
                    for label_id, i, j in zip(preds, indx_i, indx_j)]
      gold_spans = [[self.rev_tag_dict[label_id], int(i), int(j)]
                    for label_id, i, j in zip(golds, indx_i, indx_j)]

      ####################
      # Write the result #
      ####################
      fout_hdf5.create_dataset(
        name='{}'.format(valid_sent_id),
        dtype='float32',
        data=span_reps)
      results.append({"sent_id": valid_sent_id,
                      "words": record["words"],
                      "gold_spans": gold_spans,
                      "pred_spans": pred_spans})
    fout_hdf5.close()
    write_json(os.path.join(self.cfg["checkpoint_path"],
                            "%s.spans.json" % data_name), results)
    self.logger.info(
      "-- Time: %f seconds\nFINISHED." % (time.time() - start_time))


class MaskSpanModel(SpanModel):

  def _add_placeholders(self):
    self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
    self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")
    self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
    self.masks = tf.placeholder(tf.float32, shape=[None, None], name="mask")
    if self.cfg["use_chars"]:
      self.chars = tf.placeholder(tf.int32, shape=[None, None, None],
                                  name="chars")
    # hyperparameters
    self.is_train = tf.placeholder(tf.bool, name="is_train")
    self.keep_prob = tf.placeholder(tf.float32, name="rnn_keep_probability")
    self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.lr = tf.placeholder(tf.float32, name="learning_rate")

  def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
    feed_dict = {self.words: batch["words"],
                 self.seq_len: batch["seq_len"],
                 self.masks: batch["masks"]}
    if "tags" in batch:
      feed_dict[self.tags] = batch["tags"]
    if self.cfg["use_chars"]:
      feed_dict[self.chars] = batch["chars"]
    feed_dict[self.keep_prob] = keep_prob
    feed_dict[self.drop_rate] = 1.0 - keep_prob
    feed_dict[self.is_train] = is_train
    if lr is not None:
      feed_dict[self.lr] = lr
    return feed_dict

  def _build_rnn_op(self):
    with tf.variable_scope("bi_directional_rnn"):
      cell_fw = self._create_rnn_cell()
      cell_bw = self._create_rnn_cell()

      if self.cfg["use_stack_rnn"]:
        rnn_outs, *_ = stack_bidirectional_dynamic_rnn(
          cell_fw, cell_bw, self.word_emb,
          dtype=tf.float32, sequence_length=self.seq_len)
      else:
        rnn_outs, *_ = bidirectional_dynamic_rnn(
          cell_fw, cell_bw, self.word_emb,
          dtype=tf.float32, sequence_length=self.seq_len)
      rnn_outs = tf.concat(rnn_outs, axis=-1)
      rnn_outs = tf.layers.dropout(rnn_outs,
                                   rate=self.drop_rate,
                                   training=self.is_train)
      self.rnn_outs = rnn_outs
      print("rnn output shape: {}".format(rnn_outs.get_shape().as_list()))

  def _build_loss_op(self):
    self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.logits, labels=self.tags)
    self.losses = self.losses * self.masks
    self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, axis=-1))
    tf.summary.scalar("loss", self.loss)

  def _build_predict_op(self):
    self.predicts = tf.cast(tf.argmax(self.logits, axis=-1),
                            tf.int32) * tf.cast(self.masks, tf.int32)
