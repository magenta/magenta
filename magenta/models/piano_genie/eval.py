# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Piano Genie continuous eval script."""
import collections
import os
import time

from magenta.models.piano_genie import gold
from magenta.models.piano_genie.configs import get_named_config
from magenta.models.piano_genie.loader import load_noteseqs
from magenta.models.piano_genie.model import build_genie_model
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_fp", "./data/valid*.tfrecord",
                    "Path to dataset containing TFRecords of NoteSequences.")
flags.DEFINE_string("train_dir", "", "The directory for this experiment.")
flags.DEFINE_string("eval_dir", "", "The directory for evaluation output.")
flags.DEFINE_string("model_cfg", "piano_genie_paper",
                    "Hyperparameter configuration.")
flags.DEFINE_string("model_cfg_overrides", "",
                    "E.g. rnn_nlayers=4,rnn_nunits=256")
flags.DEFINE_string("ckpt_fp", None,
                    "If specified, only evaluate a single checkpoint.")


def main(unused_argv):
  if not tf.gfile.IsDirectory(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)

  cfg, _ = get_named_config(FLAGS.model_cfg, FLAGS.model_cfg_overrides)

  # Load data
  with tf.name_scope("loader"):
    feat_dict = load_noteseqs(
        FLAGS.dataset_fp,
        cfg.eval_batch_size,
        cfg.eval_seq_len,
        max_discrete_times=cfg.data_max_discrete_times,
        max_discrete_velocities=cfg.data_max_discrete_velocities,
        augment_stretch_bounds=None,
        augment_transpose_bounds=None,
        randomize_chord_order=cfg.data_randomize_chord_order,
        repeat=False)

  # Build model
  with tf.variable_scope("phero_model"):
    model_dict = build_genie_model(
        feat_dict,
        cfg,
        cfg.eval_batch_size,
        cfg.eval_seq_len,
        is_training=False)
  genie_vars = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, scope="phero_model")

  # Build gold model
  eval_gold = False
  if cfg.stp_emb_vq or cfg.stp_emb_iq:
    eval_gold = True
    with tf.variable_scope("phero_model", reuse=True):
      gold_feat_dict = {
          "midi_pitches": tf.placeholder(tf.int32, [1, None]),
          "velocities": tf.placeholder(tf.int32, [1, None]),
          "delta_times_int": tf.placeholder(tf.int32, [1, None])
      }
      gold_seq_maxlen = gold.gold_longest()
      gold_seq_varlens = tf.placeholder(tf.int32, [1])
      gold_buttons = tf.placeholder(tf.int32, [1, None])
      gold_model_dict = build_genie_model(
          gold_feat_dict,
          cfg,
          1,
          gold_seq_maxlen,
          is_training=False,
          seq_varlens=gold_seq_varlens)

    gold_encodings = gold_model_dict[
        "stp_emb_vq_discrete" if cfg.stp_emb_vq else "stp_emb_iq_discrete"]
    gold_mask = tf.sequence_mask(
        gold_seq_varlens, maxlen=gold_seq_maxlen, dtype=tf.float32)
    gold_diff = tf.cast(gold_buttons, tf.float32) - tf.cast(
        gold_encodings, tf.float32)
    gold_diff_l2 = tf.square(gold_diff)
    gold_diff_l1 = tf.abs(gold_diff)

    weighted_avg = lambda t, m: tf.reduce_sum(t * m) / tf.reduce_sum(m)

    gold_diff_l2 = weighted_avg(gold_diff_l2, gold_mask)
    gold_diff_l1 = weighted_avg(gold_diff_l1, gold_mask)

    gold_diff_l2_placeholder = tf.placeholder(tf.float32, [None])
    gold_diff_l1_placeholder = tf.placeholder(tf.float32, [None])

  summary_name_to_batch_tensor = {}

  # Summarize quantized step embeddings
  if cfg.stp_emb_vq:
    summary_name_to_batch_tensor["codebook_perplexity"] = model_dict[
        "stp_emb_vq_codebook_ppl"]
    summary_name_to_batch_tensor["loss_vqvae"] = model_dict["stp_emb_vq_loss"]

  # Summarize integer-quantized step embeddings
  if cfg.stp_emb_iq:
    summary_name_to_batch_tensor["discrete_perplexity"] = model_dict[
        "stp_emb_iq_discrete_ppl"]
    summary_name_to_batch_tensor["iq_valid_p"] = model_dict[
        "stp_emb_iq_valid_p"]
    summary_name_to_batch_tensor["loss_iq_range"] = model_dict[
        "stp_emb_iq_range_penalty"]
    summary_name_to_batch_tensor["loss_iq_contour"] = model_dict[
        "stp_emb_iq_contour_penalty"]
    summary_name_to_batch_tensor["loss_iq_deviate"] = model_dict[
        "stp_emb_iq_deviate_penalty"]

  if cfg.stp_emb_vq or cfg.stp_emb_iq:
    summary_name_to_batch_tensor["contour_violation"] = model_dict[
        "contour_violation"]
    summary_name_to_batch_tensor["deviate_violation"] = model_dict[
        "deviate_violation"]

  # Summarize VAE sequence embeddings
  if cfg.seq_emb_vae:
    summary_name_to_batch_tensor["loss_kl"] = model_dict["seq_emb_vae_kl"]

  # Reconstruction loss
  summary_name_to_batch_tensor["loss_recons"] = model_dict["dec_recons_loss"]
  summary_name_to_batch_tensor["ppl_recons"] = tf.exp(
      model_dict["dec_recons_loss"])
  if cfg.dec_pred_velocity:
    summary_name_to_batch_tensor["loss_recons_velocity"] = model_dict[
        "dec_recons_velocity_loss"]
    summary_name_to_batch_tensor["ppl_recons_velocity"] = tf.exp(
        model_dict["dec_recons_velocity_loss"])

  # Create dataset summaries
  summaries = []
  summary_name_to_placeholder = {}
  for name in summary_name_to_batch_tensor:
    placeholder = tf.placeholder(tf.float32, [None])
    summary_name_to_placeholder[name] = placeholder
    summaries.append(tf.summary.scalar(name, tf.reduce_mean(placeholder)))
  if eval_gold:
    summary_name_to_placeholder["gold_diff_l2"] = gold_diff_l2_placeholder
    summaries.append(
        tf.summary.scalar("gold_diff_l2",
                          tf.reduce_mean(gold_diff_l2_placeholder)))
    summary_name_to_placeholder["gold_diff_l1"] = gold_diff_l1_placeholder
    summaries.append(
        tf.summary.scalar("gold_diff_l1",
                          tf.reduce_mean(gold_diff_l1_placeholder)))

  summaries = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  # Create saver
  step = tf.train.get_or_create_global_step()
  saver = tf.train.Saver(genie_vars + [step], max_to_keep=None)

  def _eval_all(sess):
    """Gathers all metrics for a ckpt."""
    summaries = collections.defaultdict(list)

    if eval_gold:
      for midi_notes, buttons, seq_varlen in gold.gold_iterator([-6, 6]):
        gold_diff_l1_seq, gold_diff_l2_seq = sess.run(
            [gold_diff_l1, gold_diff_l2], {
                gold_feat_dict["midi_pitches"]:
                    midi_notes,
                gold_feat_dict["delta_times_int"]:
                    np.ones_like(midi_notes) * 8,
                gold_seq_varlens: [seq_varlen],
                gold_buttons: buttons
            })
        summaries["gold_diff_l1"].append(gold_diff_l1_seq)
        summaries["gold_diff_l2"].append(gold_diff_l2_seq)

    while True:
      try:
        batches = sess.run(summary_name_to_batch_tensor)
      except tf.errors.OutOfRangeError:
        break

      for name, scalar in batches.items():
        summaries[name].append(scalar)

    return summaries

  # Eval
  if FLAGS.ckpt_fp is None:
    ckpt_fp = None
    while True:
      latest_ckpt_fp = tf.train.latest_checkpoint(FLAGS.train_dir)

      if latest_ckpt_fp != ckpt_fp:
        print("Eval: {}".format(latest_ckpt_fp))

        with tf.Session() as sess:
          sess.run(tf.local_variables_initializer())
          saver.restore(sess, latest_ckpt_fp)

          ckpt_summaries = _eval_all(sess)
          ckpt_summaries, ckpt_step = sess.run(
              [summaries, step],
              feed_dict={
                  summary_name_to_placeholder[n]: v
                  for n, v in ckpt_summaries.items()
              })
          summary_writer.add_summary(ckpt_summaries, ckpt_step)

          saver.save(
              sess, os.path.join(FLAGS.eval_dir, "ckpt"), global_step=ckpt_step)

        print("Done")
        ckpt_fp = latest_ckpt_fp

      time.sleep(1)
  else:
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      saver.restore(sess, FLAGS.ckpt_fp)

      ckpt_summaries = _eval_all(sess)
      ckpt_step = sess.run(step)

      print("-" * 80)
      print("Ckpt: {}".format(FLAGS.ckpt_fp))
      print("Step: {}".format(ckpt_step))
      for n, l in sorted(list(ckpt_summaries.items()), key=lambda x: x[0]):
        print("{}: {}".format(n, np.mean(l)))


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()
