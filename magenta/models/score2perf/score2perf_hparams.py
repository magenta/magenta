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

"""Hparams for symbolic music modeling tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer
from tensor2tensor.utils import registry


def update_transformer_hparams_for_music(hparams):
  """Updates hparams for symbolic music modeling problems."""
  hparams.shared_embedding_and_softmax_weights = False
  hparams.symbol_modality_num_shards = 1
  hparams.label_smoothing = 0.0

  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1

  hparams.max_length = 0
  hparams.batch_size = 2048

  hparams.sampling_method = "random"
  hparams.summarize_vars = True

  # For local attention.
  hparams.add_hparam("block_length", 512)

  # For decoding.
  hparams.add_hparam("decode_output_dir", "/tmp/")

  # For multi-embedding aggregation in melody & performance autoencoder
  hparams.add_hparam("aggregation", "sum")


def update_truncate_length(hparams, length):
  hparams.max_target_seq_length = length
  hparams.min_length = length
  # If using relative attention, set max relative distance to be half
  # the max training length.
  hparams.max_relative_position = int(length / 2.0)


def update_dropout(hparams, dropout):
  """Updates dropout rate."""
  hparams.layer_prepostprocess_dropout = dropout
  hparams.attention_dropout = dropout
  hparams.relu_dropout = dropout


def update_tiny(hparams):
  hparams.hidden_size = 256
  hparams.attention_key_channels = 512
  hparams.filter_size = 2048

  hparams.batch_size = 9000
  # did not actually adjust number of warmup steps
  # hparams.warmup = 10000
  hparams.learning_rate = 0.1
  return hparams


def update_small(hparams):
  hparams.hidden_size = 384
  hparams.attention_key_channels = 512
  hparams.filter_size = 1024


def update_small_lr(hparams):
  hparams.learning_rate = 0.1
  hparams.hidden_size = 384
  hparams.attention_key_channels = 512
  hparams.filter_size = 1024


def update_medium(hparams):
  hparams.learning_rate = 0.1
  hparams.hidden_size = 512
  hparams.attention_key_channels = 512
  hparams.filter_size = 1024


#============== n8 d20 =================
@registry.register_hparams
def t_rel_len2048_h384_att512_fs1024_n8_dropout20():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  update_dropout(hparams, 0.20)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def t_len2048_h384_att512_fs1024_n8_dropout20():
  """Hparams for LM with regular attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  update_dropout(hparams, 0.20)
  hparams.num_hidden_layers = 8
  return hparams


#============= d10 ==================
@registry.register_hparams
def t_rel_len2048_h512_att512_fs1024_n6_dropout10():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_medium(hparams)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_hparams
def t_rel_len2048_h384_att512_fs1024_n6_dropout10():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small_lr(hparams)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_hparams
def t_rel_len2048_h512_att512_fs1024_n8_dropout10():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_medium(hparams)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def t_rel_len2048_h384_att512_fs1024_n8_dropout10():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 8
  return hparams


#============== d15 =================
@registry.register_hparams
def t_rel_len2048_h384_att512_fs1024_n6_dropout15():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  update_dropout(hparams, 0.15)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_hparams
def t_rel_len1024_h384_att512_fs1024_n2_dropout15():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 1024)
  update_small(hparams)
  update_dropout(hparams, 0.15)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_hparams
def t_rel_len2048_h384_att512_fs1024_n8_dropout15():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  update_dropout(hparams, 0.15)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def t_len2048_h384_att512_fs1024_n8_dropout15():
  """Hparams for LM with regular attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  update_dropout(hparams, 0.15)
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def t_rel_len2048_h384_att512_fs1024_n10_dropout30():
  """Hparams for LM with relative attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  update_dropout(hparams, 0.3)
  hparams.self_attention_type = "dot_product_relative_v2"
  hparams.num_hidden_layers = 10
  return hparams


@registry.register_hparams
def t_len2048_h384_att512_fs1024_n10_dropout30():
  """Hparams for LM with regular attention."""
  hparams = transformer.transformer_base()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_small(hparams)
  update_dropout(hparams, 0.3)
  hparams.num_hidden_layers = 10
  return hparams


@registry.register_hparams
def t_rel_len2048_dropout15_tiny():
  """Hparams for LM with relative attention, tiny transformer."""
  # hparams = transformer.transformer_base()
  hparams = transformer.transformer_tiny()
  update_transformer_hparams_for_music(hparams)
  update_truncate_length(hparams, 2048)
  update_dropout(hparams, 0.15)
  hparams.self_attention_type = "dot_product_relative_v2"
  # Need to specify num_hidden_layers
  hparams.attention_key_channels = 512
  hparams.num_hidden_layers = 8
  return hparams
