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
"""Constructs a Piano Genie model."""

from magenta.contrib import rnn as contrib_rnn
from magenta.models.piano_genie import util
import tensorflow.compat.v1 as tf

rnn = tf.nn.rnn_cell


def simple_lstm_encoder(features,
                        seq_lens,
                        rnn_celltype="lstm",
                        rnn_nlayers=2,
                        rnn_nunits=128,
                        rnn_bidirectional=True,
                        dtype=tf.float32):
  """Constructs an LSTM-based encoder."""
  x = features

  with tf.variable_scope("rnn_input"):
    x = tf.layers.dense(x, rnn_nunits)

  if rnn_celltype == "lstm":
    celltype = contrib_rnn.LSTMBlockCell
  else:
    raise NotImplementedError()

  cell = rnn.MultiRNNCell(
      [celltype(rnn_nunits) for _ in range(rnn_nlayers)])

  with tf.variable_scope("rnn"):
    if rnn_bidirectional:
      (x_fw, x_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell,
          cell_bw=cell,
          inputs=x,
          sequence_length=seq_lens,
          dtype=dtype)
      x = tf.concat([x_fw, x_bw], axis=2)

      state_fw, state_bw = state_fw[-1].h, state_bw[-1].h
      state = tf.concat([state_fw, state_bw], axis=1)
    else:
      # initial_state = cell.zero_state(batch_size, dtype)
      x, state = tf.nn.dynamic_rnn(
          cell=cell, inputs=x, sequence_length=seq_lens, dtype=dtype)
      state = state[-1].h

  return x, state


def simple_lstm_decoder(features,
                        seq_lens,
                        batch_size,
                        rnn_celltype="lstm",
                        rnn_nlayers=2,
                        rnn_nunits=128,
                        dtype=tf.float32):
  """Constructs an LSTM-based decoder."""
  x = features

  with tf.variable_scope("rnn_input"):
    x = tf.layers.dense(x, rnn_nunits)

  if rnn_celltype == "lstm":
    celltype = contrib_rnn.LSTMBlockCell
  else:
    raise NotImplementedError()

  cell = rnn.MultiRNNCell(
      [celltype(rnn_nunits) for _ in range(rnn_nlayers)])

  with tf.variable_scope("rnn"):
    initial_state = cell.zero_state(batch_size, dtype)
    x, final_state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=x,
        sequence_length=seq_lens,
        initial_state=initial_state)

  return x, initial_state, final_state


def weighted_avg(t, mask=None, axis=None, expand_mask=False):
  if mask is None:
    return tf.reduce_mean(t, axis=axis)
  else:
    if expand_mask:
      mask = tf.expand_dims(mask, axis=-1)
    return tf.reduce_sum(
        tf.multiply(t, mask), axis=axis) / tf.reduce_sum(
            mask, axis=axis)


def build_genie_model(feat_dict,
                      cfg,
                      batch_size,
                      seq_len,
                      is_training=True,
                      seq_varlens=None,
                      dtype=tf.float32):
  """Builds a Piano Genie model.

  Args:
    feat_dict: Dictionary containing input tensors.
    cfg: Configuration object.
    batch_size: Number of items in batch.
    seq_len: Length of each batch item.
    is_training: Set to False for evaluation.
    seq_varlens: If not None, a tensor with the batch sequence lengths.
    dtype: Model weight type.

  Returns:
    A dict containing tensors for relevant model config.
  """
  out_dict = {}

  # Parse features
  pitches = util.demidify(feat_dict["midi_pitches"])
  velocities = feat_dict["velocities"]
  pitches_scalar = ((tf.cast(pitches, tf.float32) / 87.) * 2.) - 1.

  # Create sequence lens
  if is_training and cfg.train_randomize_seq_len:
    seq_lens = tf.random_uniform(
        [batch_size],
        minval=cfg.train_seq_len_min,
        maxval=seq_len + 1,
        dtype=tf.int32)
    stp_varlen_mask = tf.sequence_mask(
        seq_lens, maxlen=seq_len, dtype=tf.float32)
  elif seq_varlens is not None:
    seq_lens = seq_varlens
    stp_varlen_mask = tf.sequence_mask(
        seq_varlens, maxlen=seq_len, dtype=tf.float32)
  else:
    seq_lens = tf.ones([batch_size], dtype=tf.int32) * seq_len
    stp_varlen_mask = None

  # Encode
  if (cfg.stp_emb_unconstrained or cfg.stp_emb_vq or cfg.stp_emb_iq or
      cfg.seq_emb_unconstrained or cfg.seq_emb_vae or
      cfg.lor_emb_unconstrained):
    # Build encoder features
    enc_feats = []
    if cfg.enc_pitch_scalar:
      enc_feats.append(tf.expand_dims(pitches_scalar, axis=-1))
    else:
      enc_feats.append(tf.one_hot(pitches, 88))
    if "delta_times_int" in cfg.enc_aux_feats:
      enc_feats.append(
          tf.one_hot(feat_dict["delta_times_int"],
                     cfg.data_max_discrete_times + 1))
    if "velocities" in cfg.enc_aux_feats:
      enc_feats.append(
          tf.one_hot(velocities, cfg.data_max_discrete_velocities + 1))
    enc_feats = tf.concat(enc_feats, axis=2)

    with tf.variable_scope("encoder"):
      enc_stp, enc_seq = simple_lstm_encoder(
          enc_feats,
          seq_lens,
          rnn_celltype=cfg.rnn_celltype,
          rnn_nlayers=cfg.rnn_nlayers,
          rnn_nunits=cfg.rnn_nunits,
          rnn_bidirectional=cfg.enc_rnn_bidirectional,
          dtype=dtype)

  latents = []

  # Step embeddings (single vector per timestep)
  if cfg.stp_emb_unconstrained:
    with tf.variable_scope("stp_emb_unconstrained"):
      stp_emb_unconstrained = tf.layers.dense(
          enc_stp, cfg.stp_emb_unconstrained_embedding_dim)

    out_dict["stp_emb_unconstrained"] = stp_emb_unconstrained
    latents.append(stp_emb_unconstrained)

  # Quantized step embeddings with VQ-VAE
  if cfg.stp_emb_vq:
    import sonnet as snt  # pylint:disable=g-import-not-at-top,import-outside-toplevel
    with tf.variable_scope("stp_emb_vq"):
      with tf.variable_scope("pre_vq"):
        # pre_vq_encoding is tf.float32 of [batch_size, seq_len, embedding_dim]
        pre_vq_encoding = tf.layers.dense(enc_stp, cfg.stp_emb_vq_embedding_dim)

      with tf.variable_scope("quantizer"):
        assert stp_varlen_mask is None
        vq_vae = snt.nets.VectorQuantizer(
            embedding_dim=cfg.stp_emb_vq_embedding_dim,
            num_embeddings=cfg.stp_emb_vq_codebook_size,
            commitment_cost=cfg.stp_emb_vq_commitment_cost)
        vq_vae_output = vq_vae(pre_vq_encoding, is_training=is_training)

        stp_emb_vq_quantized = vq_vae_output["quantize"]
        stp_emb_vq_discrete = tf.reshape(
            tf.argmax(vq_vae_output["encodings"], axis=1, output_type=tf.int32),
            [batch_size, seq_len])
        stp_emb_vq_codebook = tf.transpose(vq_vae.embeddings)

    out_dict["stp_emb_vq_quantized"] = stp_emb_vq_quantized
    out_dict["stp_emb_vq_discrete"] = stp_emb_vq_discrete
    out_dict["stp_emb_vq_loss"] = vq_vae_output["loss"]
    out_dict["stp_emb_vq_codebook"] = stp_emb_vq_codebook
    out_dict["stp_emb_vq_codebook_ppl"] = vq_vae_output["perplexity"]
    latents.append(stp_emb_vq_quantized)

    # This tensor retrieves continuous embeddings from codebook. It should
    # *never* be used during training.
    out_dict["stp_emb_vq_quantized_lookup"] = tf.nn.embedding_lookup(
        stp_emb_vq_codebook, stp_emb_vq_discrete)

  # Integer-quantized step embeddings with straight-through
  if cfg.stp_emb_iq:
    with tf.variable_scope("stp_emb_iq"):
      with tf.variable_scope("pre_iq"):
        # pre_iq_encoding is tf.float32 of [batch_size, seq_len]
        pre_iq_encoding = tf.layers.dense(enc_stp, 1)[:, :, 0]

      def iqst(x, n):
        """Integer quantization with straight-through estimator."""
        eps = 1e-7
        s = float(n - 1)
        xp = tf.clip_by_value((x + 1) / 2.0, -eps, 1 + eps)
        xpp = tf.round(s * xp)
        xppp = 2 * (xpp / s) - 1
        return xpp, x + tf.stop_gradient(xppp - x)

      with tf.variable_scope("quantizer"):
        # Pass rounded vals to decoder w/ straight-through estimator
        stp_emb_iq_discrete_f, stp_emb_iq_discrete_rescaled = iqst(
            pre_iq_encoding, cfg.stp_emb_iq_nbins)
        stp_emb_iq_discrete = tf.cast(stp_emb_iq_discrete_f + 1e-4, tf.int32)
        stp_emb_iq_discrete_f = tf.cast(stp_emb_iq_discrete, tf.float32)
        stp_emb_iq_quantized = tf.expand_dims(
            stp_emb_iq_discrete_rescaled, axis=2)

        # Determine which elements round to valid indices
        stp_emb_iq_inrange = tf.logical_and(
            tf.greater_equal(pre_iq_encoding, -1),
            tf.less_equal(pre_iq_encoding, 1))
        stp_emb_iq_inrange_mask = tf.cast(stp_emb_iq_inrange, tf.float32)
        stp_emb_iq_valid_p = weighted_avg(stp_emb_iq_inrange_mask,
                                          stp_varlen_mask)

        # Regularize to encourage encoder to output in range
        stp_emb_iq_range_penalty = weighted_avg(
            tf.square(tf.maximum(tf.abs(pre_iq_encoding) - 1, 0)),
            stp_varlen_mask)

        # Regularize to correlate latent finite differences to input
        stp_emb_iq_dlatents = pre_iq_encoding[:, 1:] - pre_iq_encoding[:, :-1]
        if cfg.stp_emb_iq_contour_dy_scalar:
          stp_emb_iq_dnotes = pitches_scalar[:, 1:] - pitches_scalar[:, :-1]
        else:
          stp_emb_iq_dnotes = tf.cast(pitches[:, 1:] - pitches[:, :-1],
                                      tf.float32)
        if cfg.stp_emb_iq_contour_exp == 1:
          power_func = tf.identity
        elif cfg.stp_emb_iq_contour_exp == 2:
          power_func = tf.square
        else:
          raise NotImplementedError()
        if cfg.stp_emb_iq_contour_comp == "product":
          comp_func = tf.multiply
        elif cfg.stp_emb_iq_contour_comp == "quotient":
          comp_func = lambda x, y: tf.divide(x, y + 1e-6)
        else:
          raise NotImplementedError()

        stp_emb_iq_contour_penalty = weighted_avg(
            power_func(
                tf.maximum(
                    cfg.stp_emb_iq_contour_margin - comp_func(
                        stp_emb_iq_dnotes, stp_emb_iq_dlatents), 0)),
            None if stp_varlen_mask is None else stp_varlen_mask[:, 1:])

        # Regularize to maintain note consistency
        stp_emb_iq_note_held = tf.cast(
            tf.equal(pitches[:, 1:] - pitches[:, :-1], 0), tf.float32)
        if cfg.stp_emb_iq_deviate_exp == 1:
          power_func = tf.abs
        elif cfg.stp_emb_iq_deviate_exp == 2:
          power_func = tf.square

        if stp_varlen_mask is None:
          mask = stp_emb_iq_note_held
        else:
          mask = stp_varlen_mask[:, 1:] * stp_emb_iq_note_held
        stp_emb_iq_deviate_penalty = weighted_avg(
            power_func(stp_emb_iq_dlatents), mask)

        # Calculate perplexity of discrete encoder posterior
        if stp_varlen_mask is None:
          mask = stp_emb_iq_inrange_mask
        else:
          mask = stp_varlen_mask * stp_emb_iq_inrange_mask
        stp_emb_iq_discrete_oh = tf.one_hot(stp_emb_iq_discrete,
                                            cfg.stp_emb_iq_nbins)
        stp_emb_iq_avg_probs = weighted_avg(
            stp_emb_iq_discrete_oh,
            mask,
            axis=[0, 1],
            expand_mask=True)
        stp_emb_iq_discrete_ppl = tf.exp(-tf.reduce_sum(
            stp_emb_iq_avg_probs * tf.log(stp_emb_iq_avg_probs + 1e-10)))

    out_dict["stp_emb_iq_quantized"] = stp_emb_iq_quantized
    out_dict["stp_emb_iq_discrete"] = stp_emb_iq_discrete
    out_dict["stp_emb_iq_valid_p"] = stp_emb_iq_valid_p
    out_dict["stp_emb_iq_range_penalty"] = stp_emb_iq_range_penalty
    out_dict["stp_emb_iq_contour_penalty"] = stp_emb_iq_contour_penalty
    out_dict["stp_emb_iq_deviate_penalty"] = stp_emb_iq_deviate_penalty
    out_dict["stp_emb_iq_discrete_ppl"] = stp_emb_iq_discrete_ppl
    latents.append(stp_emb_iq_quantized)

    # This tensor converts discrete values to continuous.
    # It should *never* be used during training.
    out_dict["stp_emb_iq_quantized_lookup"] = tf.expand_dims(
        2. * (stp_emb_iq_discrete_f / (cfg.stp_emb_iq_nbins - 1.)) - 1., axis=2)

  # Sequence embedding (single vector per sequence)
  if cfg.seq_emb_unconstrained:
    with tf.variable_scope("seq_emb_unconstrained"):
      seq_emb_unconstrained = tf.layers.dense(
          enc_seq, cfg.seq_emb_unconstrained_embedding_dim)

    out_dict["seq_emb_unconstrained"] = seq_emb_unconstrained

    seq_emb_unconstrained = tf.stack([seq_emb_unconstrained] * seq_len, axis=1)
    latents.append(seq_emb_unconstrained)

  # Sequence embeddings (variational w/ reparameterization trick)
  if cfg.seq_emb_vae:
    with tf.variable_scope("seq_emb_vae"):
      seq_emb_vae = tf.layers.dense(enc_seq, cfg.seq_emb_vae_embedding_dim * 2)

      mean = seq_emb_vae[:, :cfg.seq_emb_vae_embedding_dim]
      stddev = 1e-6 + tf.nn.softplus(
          seq_emb_vae[:, cfg.seq_emb_vae_embedding_dim:])
      seq_emb_vae = mean + stddev * tf.random_normal(
          tf.shape(mean), 0, 1, dtype=dtype)

      kl = tf.reduce_mean(0.5 * tf.reduce_sum(
          tf.square(mean) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev))
          - 1,
          axis=1))

    out_dict["seq_emb_vae"] = seq_emb_vae
    out_dict["seq_emb_vae_kl"] = kl

    seq_emb_vae = tf.stack([seq_emb_vae] * seq_len, axis=1)
    latents.append(seq_emb_vae)

  # Low-rate embeddings
  if cfg.lor_emb_unconstrained:
    assert seq_len % cfg.lor_emb_n == 0

    with tf.variable_scope("lor_emb_unconstrained"):
      # Downsample step embeddings
      rnn_embedding_dim = int(enc_stp.get_shape()[-1])
      enc_lor = tf.reshape(enc_stp, [
          batch_size, seq_len // cfg.lor_emb_n,
          cfg.lor_emb_n * rnn_embedding_dim
      ])
      lor_emb_unconstrained = tf.layers.dense(
          enc_lor, cfg.lor_emb_unconstrained_embedding_dim)

      out_dict["lor_emb_unconstrained"] = lor_emb_unconstrained

      # Upsample lo-rate embeddings for decoding
      lor_emb_unconstrained = tf.expand_dims(lor_emb_unconstrained, axis=2)
      lor_emb_unconstrained = tf.tile(lor_emb_unconstrained,
                                      [1, 1, cfg.lor_emb_n, 1])
      lor_emb_unconstrained = tf.reshape(
          lor_emb_unconstrained,
          [batch_size, seq_len, cfg.lor_emb_unconstrained_embedding_dim])

      latents.append(lor_emb_unconstrained)

  # Build decoder features
  dec_feats = latents

  if cfg.dec_autoregressive:
    # Retrieve pitch numbers
    curr_pitches = pitches
    last_pitches = curr_pitches[:, :-1]
    last_pitches = tf.pad(
        last_pitches, [[0, 0], [1, 0]],
        constant_values=-1)  # Prepend <SOS> token
    out_dict["dec_last_pitches"] = last_pitches
    dec_feats.append(tf.one_hot(last_pitches + 1, 89))

    if cfg.dec_pred_velocity:
      curr_velocities = velocities
      last_velocities = curr_velocities[:, :-1]
      last_velocities = tf.pad(last_velocities, [[0, 0], [1, 0]])
      dec_feats.append(
          tf.one_hot(last_velocities, cfg.data_max_discrete_velocities + 1))

  if "delta_times_int" in cfg.dec_aux_feats:
    dec_feats.append(
        tf.one_hot(feat_dict["delta_times_int"],
                   cfg.data_max_discrete_times + 1))
  if "velocities" in cfg.dec_aux_feats:
    assert not cfg.dec_pred_velocity
    dec_feats.append(
        tf.one_hot(feat_dict["velocities"],
                   cfg.data_max_discrete_velocities + 1))

  assert dec_feats
  dec_feats = tf.concat(dec_feats, axis=2)

  # Decode
  with tf.variable_scope("decoder"):
    dec_stp, dec_initial_state, dec_final_state = simple_lstm_decoder(
        dec_feats,
        seq_lens,
        batch_size,
        rnn_celltype=cfg.rnn_celltype,
        rnn_nlayers=cfg.rnn_nlayers,
        rnn_nunits=cfg.rnn_nunits)

    with tf.variable_scope("pitches"):
      dec_recons_logits = tf.layers.dense(dec_stp, 88)

    dec_recons_loss = weighted_avg(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=dec_recons_logits, labels=pitches), stp_varlen_mask)

    out_dict["dec_initial_state"] = dec_initial_state
    out_dict["dec_final_state"] = dec_final_state
    out_dict["dec_recons_logits"] = dec_recons_logits
    out_dict["dec_recons_scores"] = tf.nn.softmax(dec_recons_logits, axis=-1)
    out_dict["dec_recons_preds"] = tf.argmax(
        dec_recons_logits, output_type=tf.int32, axis=-1)
    out_dict["dec_recons_midi_preds"] = util.remidify(
        out_dict["dec_recons_preds"])
    out_dict["dec_recons_loss"] = dec_recons_loss

    if cfg.dec_pred_velocity:
      with tf.variable_scope("velocities"):
        dec_recons_velocity_logits = tf.layers.dense(
            dec_stp, cfg.data_max_discrete_velocities + 1)

      dec_recons_velocity_loss = weighted_avg(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=dec_recons_velocity_logits, labels=velocities),
          stp_varlen_mask)

      out_dict["dec_recons_velocity_logits"] = dec_recons_velocity_logits
      out_dict["dec_recons_velocity_loss"] = dec_recons_velocity_loss

  # Stats
  if cfg.stp_emb_vq or cfg.stp_emb_iq:
    discrete = out_dict[
        "stp_emb_vq_discrete" if cfg.stp_emb_vq else "stp_emb_iq_discrete"]
    dx = pitches[:, 1:] - pitches[:, :-1]
    dy = discrete[:, 1:] - discrete[:, :-1]
    contour_violation = tf.reduce_mean(tf.cast(tf.less(dx * dy, 0), tf.float32))

    dx_hold = tf.equal(dx, 0)
    deviate_violation = weighted_avg(
        tf.cast(tf.not_equal(dy, 0), tf.float32), tf.cast(dx_hold, tf.float32))

    out_dict["contour_violation"] = contour_violation
    out_dict["deviate_violation"] = deviate_violation

  return out_dict
