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

# Lint as: python2, python3
"""Hyperparameter configurations for Piano Genie."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six


class BasePianoGenieConfig(object):
  """Base class for model configurations."""

  def __init__(self):
    # Data parameters
    self.data_max_discrete_times = 32
    self.data_max_discrete_velocities = 16
    self.data_randomize_chord_order = False

    # RNN parameters (encoder and decoder)
    self.rnn_celltype = "lstm"
    self.rnn_nlayers = 2
    self.rnn_nunits = 128

    # Encoder parameters
    self.enc_rnn_bidirectional = True
    self.enc_pitch_scalar = False
    self.enc_aux_feats = []

    # Decoder parameters
    self.dec_autoregressive = False
    self.dec_aux_feats = []
    self.dec_pred_velocity = False

    # Unconstrained "discretization" parameters
    # Passes sequence of continuous embeddings directly to decoder (which we
    # will discretize during post-processing i.e. with K-means)
    self.stp_emb_unconstrained = False
    self.stp_emb_unconstrained_embedding_dim = 64

    # VQ-VAE parameters
    self.stp_emb_vq = False
    self.stp_emb_vq_embedding_dim = 64
    self.stp_emb_vq_codebook_size = 8
    self.stp_emb_vq_commitment_cost = 0.25

    # Integer quant parameters
    self.stp_emb_iq = False
    self.stp_emb_iq_nbins = 8
    self.stp_emb_iq_contour_dy_scalar = False
    self.stp_emb_iq_contour_margin = 0.
    self.stp_emb_iq_contour_exp = 2
    self.stp_emb_iq_contour_comp = "product"
    self.stp_emb_iq_deviate_exp = 2

    # Unconstrained parameters... just like VAE but passed directly (no prior)
    self.seq_emb_unconstrained = False
    self.seq_emb_unconstrained_embedding_dim = 64

    # VAE parameters. Last hidden state of RNN will be projected to a summary
    # vector which will be passed to decoder with Gaussian re-parameterization.
    self.seq_emb_vae = False
    self.seq_emb_vae_embedding_dim = 64

    # (lo)w-(r)ate latents... one per every N steps of input
    self.lor_emb_n = 16
    self.lor_emb_unconstrained = False
    self.lor_emb_unconstrained_embedding_dim = 8

    # Training parameters
    self.train_batch_size = 32
    self.train_seq_len = 128
    self.train_seq_len_min = 1
    self.train_randomize_seq_len = False
    self.train_augment_stretch_bounds = (0.95, 1.05)
    self.train_augment_transpose_bounds = (-6, 6)
    self.train_loss_vq_err_scalar = 1.
    self.train_loss_iq_range_scalar = 1.
    self.train_loss_iq_contour_scalar = 1.
    self.train_loss_iq_deviate_scalar = 0.
    self.train_loss_vae_kl_scalar = 1.
    self.train_lr = 3e-4

    # Eval parameters
    self.eval_batch_size = 32
    self.eval_seq_len = 128


class StpFree(BasePianoGenieConfig):

  def __init__(self):
    super(StpFree, self).__init__()

    self.stp_emb_unconstrained = True


class StpVq(BasePianoGenieConfig):

  def __init__(self):
    super(StpVq, self).__init__()

    self.stp_emb_vq = True


class StpIq(BasePianoGenieConfig):

  def __init__(self):
    super(StpIq, self).__init__()

    self.stp_emb_iq = True


class SeqFree(BasePianoGenieConfig):

  def __init__(self):
    super(SeqFree, self).__init__()

    self.seq_emb_unconstrained = True


class SeqVae(BasePianoGenieConfig):

  def __init__(self):
    super(SeqVae, self).__init__()

    self.seq_emb_vae = True


class LorFree(BasePianoGenieConfig):

  def __init__(self):
    super(LorFree, self).__init__()

    self.lor_emb_unconstrained = True


class StpVqSeqVae(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqVae, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_vae = True


class StpVqSeqFree(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFree, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True


class StpVqLorFree(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqLorFree, self).__init__()

    self.stp_emb_vq = True
    self.lor_emb_unconstrained = True


class StpVqSeqFreeRand(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeRand, self).__init__()

    self.data_randomize_chord_order = True
    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True


class StpVqSeqFreePredvelo(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreePredvelo, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.dec_pred_velocity = True


class Auto(BasePianoGenieConfig):

  def __init__(self):
    super(Auto, self).__init__()

    self.dec_autoregressive = True


class StpVqAuto(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqAuto, self).__init__()

    self.stp_emb_vq = True
    self.dec_autoregressive = True


class StpIqAuto(BasePianoGenieConfig):

  def __init__(self):
    super(StpIqAuto, self).__init__()

    self.stp_emb_iq = True
    self.dec_autoregressive = True


class SeqVaeAuto(BasePianoGenieConfig):

  def __init__(self):
    super(SeqVaeAuto, self).__init__()

    self.seq_emb_vae = True
    self.dec_autoregressive = True


class LorFreeAuto(BasePianoGenieConfig):

  def __init__(self):
    super(LorFreeAuto, self).__init__()

    self.lor_emb_unconstrained = True
    self.dec_autoregressive = True


class StpVqSeqVaeAuto(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqVaeAuto, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_vae = True
    self.dec_autoregressive = True


class StpVqSeqFreeAuto(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeAuto, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.dec_autoregressive = True


class StpVqLorFreeAuto(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqLorFreeAuto, self).__init__()

    self.stp_emb_vq = True
    self.lor_emb_unconstrained = True
    self.dec_autoregressive = True


class StpVqSeqFreeAutoRand(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeAutoRand, self).__init__()

    self.data_randomize_chord_order = True
    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.dec_autoregressive = True


class StpVqSeqFreeAutoVarlen(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeAutoVarlen, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.dec_autoregressive = True
    self.train_seq_len_min = 32
    self.train_randomize_seq_len = True


class StpVqSeqFreeAutoPredvelo(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeAutoPredvelo, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.dec_autoregressive = True
    self.dec_pred_velocity = True


class StpVqSeqVaeAutoDt(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqVaeAutoDt, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_vae = True
    self.enc_aux_feats = ["delta_times_int"]
    self.dec_autoregressive = True
    self.dec_aux_feats = ["delta_times_int"]


class StpVqSeqFreeAutoDt(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeAutoDt, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.enc_aux_feats = ["delta_times_int"]
    self.dec_autoregressive = True
    self.dec_aux_feats = ["delta_times_int"]


class StpVqSeqVaeAutoVs(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqVaeAutoVs, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_vae = True
    self.enc_aux_feats = ["velocities"]
    self.dec_autoregressive = True
    self.dec_aux_feats = ["velocities"]


class StpVqSeqFreeAutoVs(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeAutoVs, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.enc_aux_feats = ["velocities"]
    self.dec_autoregressive = True
    self.dec_aux_feats = ["velocities"]


class StpVqSeqVaeAutoDtVs(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqVaeAutoDtVs, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_vae = True
    self.enc_aux_feats = ["delta_times_int", "velocities"]
    self.dec_autoregressive = True
    self.dec_aux_feats = ["delta_times_int", "velocities"]


class StpVqSeqFreeAutoDtVs(BasePianoGenieConfig):

  def __init__(self):
    super(StpVqSeqFreeAutoDtVs, self).__init__()

    self.stp_emb_vq = True
    self.seq_emb_unconstrained = True
    self.enc_aux_feats = ["delta_times_int", "velocities"]
    self.dec_autoregressive = True
    self.dec_aux_feats = ["delta_times_int", "velocities"]


class PianoGeniePaper(BasePianoGenieConfig):
  """Config matching Piano Genie paper."""

  def __init__(self):
    super(PianoGeniePaper, self).__init__()

    self.enc_aux_feats = ["delta_times_int"]
    self.dec_autoregressive = True
    self.dec_aux_feats = ["delta_times_int"]
    self.stp_emb_iq = True
    self.stp_emb_iq_contour_margin = 1.
    self.stp_emb_iq_deviate_exp = 1


_named_configs = {
    "stp_free": StpFree(),
    "stp_vq": StpVq(),
    "stp_iq": StpIq(),
    "seq_free": SeqFree(),
    "seq_vae": SeqVae(),
    "lor_free": LorFree(),
    "stp_vq_seq_vae": StpVqSeqVae(),
    "stp_vq_seq_free": StpVqSeqFree(),
    "stp_vq_lor_free": StpVqLorFree(),
    "stp_vq_seq_free_rand": StpVqSeqFreeRand(),
    "stp_vq_seq_free_predvelo": StpVqSeqFreePredvelo(),
    "auto_no_enc": Auto(),
    "stp_vq_auto": StpVqAuto(),
    "stp_iq_auto": StpIqAuto(),
    "seq_vae_auto": SeqVaeAuto(),
    "lor_free_auto": LorFreeAuto(),
    "stp_vq_seq_vae_auto": StpVqSeqVaeAuto(),
    "stp_vq_seq_free_auto": StpVqSeqFreeAuto(),
    "stp_vq_lor_free_auto": StpVqLorFreeAuto(),
    "stp_vq_seq_free_auto_rand": StpVqSeqFreeAutoRand(),
    "stp_vq_seq_free_auto_varlen": StpVqSeqFreeAutoVarlen(),
    "stp_vq_seq_free_auto_predvelo": StpVqSeqFreeAutoPredvelo(),
    "stp_vq_seq_vae_auto_dt": StpVqSeqVaeAutoDt(),
    "stp_vq_seq_free_auto_dt": StpVqSeqFreeAutoDt(),
    "stp_vq_seq_vae_auto_vs": StpVqSeqVaeAutoVs(),
    "stp_vq_seq_free_auto_vs": StpVqSeqFreeAutoVs(),
    "stp_vq_seq_vae_auto_dt_vs": StpVqSeqVaeAutoDtVs(),
    "stp_vq_seq_vae_free_dt_vs": StpVqSeqFreeAutoDtVs(),
    "piano_genie_paper": PianoGeniePaper(),
}


def get_named_config(name, overrides=None):
  """Instantiates a config object by name.

  Args:
    name: Config name (see _named_configs)
    overrides: Comma-separated list of overrides e.g. "a=1,b=2"

  Returns:
    cfg: The config object
    summary: Text summary of all params in config
  """
  cfg = _named_configs[name]

  if overrides is not None and overrides.strip():
    overrides = [p.split("=") for p in six.ensure_str(overrides).split(",")]
    for key, val in overrides:
      val_type = type(getattr(cfg, key))
      if val_type == bool:
        setattr(cfg, key, val in ["True", "true", "t", "1"])
      elif val_type == list:
        setattr(cfg, key, val.split(";"))
      else:
        setattr(cfg, key, val_type(val))

  summary = "\n".join([
      "{},{}".format(k, v)
      for k, v in sorted(list(vars(cfg).items()), key=lambda x: x[0])
  ])

  return cfg, summary
