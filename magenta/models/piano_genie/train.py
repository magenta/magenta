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

"""Piano Genie training script."""
import os

from magenta.models.piano_genie import util
from magenta.models.piano_genie.configs import get_named_config
from magenta.models.piano_genie.loader import load_noteseqs
from magenta.models.piano_genie.model import build_genie_model
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_fp", "./data/train*.tfrecord",
                    "Path to dataset containing TFRecords of NoteSequences.")
flags.DEFINE_string("train_dir", "", "The directory for this experiment")
flags.DEFINE_string("model_cfg", "piano_genie_paper",
                    "Hyperparameter configuration.")
flags.DEFINE_string("model_cfg_overrides", "",
                    "E.g. rnn_nlayers=4,rnn_nunits=256")
flags.DEFINE_integer("summary_every_nsecs", 60,
                     "Summarize to Tensorboard every n seconds.")


def main(unused_argv):
  if not tf.gfile.IsDirectory(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  cfg, cfg_summary = get_named_config(FLAGS.model_cfg,
                                      FLAGS.model_cfg_overrides)
  with tf.gfile.Open(os.path.join(FLAGS.train_dir, "cfg.txt"), "w") as f:
    f.write(cfg_summary)

  # Load data
  with tf.name_scope("loader"):
    feat_dict = load_noteseqs(
        FLAGS.dataset_fp,
        cfg.train_batch_size,
        cfg.train_seq_len,
        max_discrete_times=cfg.data_max_discrete_times,
        max_discrete_velocities=cfg.data_max_discrete_velocities,
        augment_stretch_bounds=cfg.train_augment_stretch_bounds,
        augment_transpose_bounds=cfg.train_augment_transpose_bounds,
        randomize_chord_order=cfg.data_randomize_chord_order,
        repeat=True)

  # Summarize data
  tf.summary.image(
      "piano_roll",
      util.discrete_to_piano_roll(util.demidify(feat_dict["midi_pitches"]), 88))

  # Build model
  with tf.variable_scope("phero_model"):
    model_dict = build_genie_model(
        feat_dict,
        cfg,
        cfg.train_batch_size,
        cfg.train_seq_len,
        is_training=True)

  # Summarize quantized step embeddings
  if cfg.stp_emb_vq:
    tf.summary.scalar("codebook_perplexity",
                      model_dict["stp_emb_vq_codebook_ppl"])
    tf.summary.image(
        "genie",
        util.discrete_to_piano_roll(
            model_dict["stp_emb_vq_discrete"],
            cfg.stp_emb_vq_codebook_size,
            dilation=max(1, 88 // cfg.stp_emb_vq_codebook_size)))
    tf.summary.scalar("loss_vqvae", model_dict["stp_emb_vq_loss"])

  # Summarize integer-quantized step embeddings
  if cfg.stp_emb_iq:
    tf.summary.scalar("discrete_perplexity",
                      model_dict["stp_emb_iq_discrete_ppl"])
    tf.summary.scalar("iq_valid_p", model_dict["stp_emb_iq_valid_p"])
    tf.summary.image(
        "genie",
        util.discrete_to_piano_roll(
            model_dict["stp_emb_iq_discrete"],
            cfg.stp_emb_iq_nbins,
            dilation=max(1, 88 // cfg.stp_emb_iq_nbins)))
    tf.summary.scalar("loss_iq_range", model_dict["stp_emb_iq_range_penalty"])
    tf.summary.scalar("loss_iq_contour",
                      model_dict["stp_emb_iq_contour_penalty"])
    tf.summary.scalar("loss_iq_deviate",
                      model_dict["stp_emb_iq_deviate_penalty"])

  if cfg.stp_emb_vq or cfg.stp_emb_iq:
    tf.summary.scalar("contour_violation", model_dict["contour_violation"])
    tf.summary.scalar("deviate_violation", model_dict["deviate_violation"])

  # Summarize VAE sequence embeddings
  if cfg.seq_emb_vae:
    tf.summary.scalar("loss_kl", model_dict["seq_emb_vae_kl"])

  # Summarize output
  tf.summary.image(
      "decoder_scores",
      util.discrete_to_piano_roll(model_dict["dec_recons_scores"], 88))
  tf.summary.image(
      "decoder_preds",
      util.discrete_to_piano_roll(model_dict["dec_recons_preds"], 88))
  if cfg.dec_pred_velocity:
    tf.summary.scalar("loss_recons_velocity",
                      model_dict["dec_recons_velocity_loss"])
    tf.summary.scalar("ppl_recons_velocity",
                      tf.exp(model_dict["dec_recons_velocity_loss"]))

  # Reconstruction loss
  tf.summary.scalar("loss_recons", model_dict["dec_recons_loss"])
  tf.summary.scalar("ppl_recons", tf.exp(model_dict["dec_recons_loss"]))

  # Build hybrid loss
  loss = model_dict["dec_recons_loss"]
  if cfg.stp_emb_vq and cfg.train_loss_vq_err_scalar > 0:
    loss += (cfg.train_loss_vq_err_scalar * model_dict["stp_emb_vq_loss"])
  if cfg.stp_emb_iq and cfg.train_loss_iq_range_scalar > 0:
    loss += (
        cfg.train_loss_iq_range_scalar * model_dict["stp_emb_iq_range_penalty"])
  if cfg.stp_emb_iq and cfg.train_loss_iq_contour_scalar > 0:
    loss += (
        cfg.train_loss_iq_contour_scalar *
        model_dict["stp_emb_iq_contour_penalty"])
  if cfg.stp_emb_iq and cfg.train_loss_iq_deviate_scalar > 0:
    loss += (
        cfg.train_loss_iq_deviate_scalar *
        model_dict["stp_emb_iq_deviate_penalty"])
  if cfg.seq_emb_vae and cfg.train_loss_vae_kl_scalar > 0:
    loss += (cfg.train_loss_vae_kl_scalar * model_dict["seq_emb_vae_kl"])
  if cfg.dec_pred_velocity:
    loss += model_dict["dec_recons_velocity_loss"]
  tf.summary.scalar("loss", loss)

  # Construct optimizer
  opt = tf.train.AdamOptimizer(learning_rate=cfg.train_lr)
  train_op = opt.minimize(
      loss, global_step=tf.train.get_or_create_global_step())

  # Train
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.train_dir,
      save_checkpoint_secs=600,
      save_summaries_secs=FLAGS.summary_every_nsecs) as sess:
    while True:
      sess.run(train_op)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.app.run()
