"""Configurations for MusicVAE models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.common import merge_hparams
from magenta.models.music_vae import data
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE
from tensorflow.contrib.training import HParams


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter',
     'note_sequence_converter', 'train_examples_path', 'eval_examples_path'])):

  def values(self):
    return self._asdict()


def update_config(config, update_dict):
  config_dict = config.values()
  config_dict.update(update_dict)
  return Config(**config_dict)


CONFIG_MAP = {}


# Melody
CONFIG_MAP['cat-mel_2bar_small'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['cat-mel_2bar_big'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)

# Drums
CONFIG_MAP['cat-drums_2bar_small'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['cat-drums_2bar_big'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['nade-drums_2bar_reduced'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.MultiLabelRnnNadeDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[1024],
            dec_rnn_size=[512, 512],
            nade_num_hidden=128,
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True,
        roll_output=True),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['nade-drums_2bar_full'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.MultiLabelRnnNadeDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[1024],
            dec_rnn_size=[512, 512],
            nade_num_hidden=128,
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        pitch_classes=data.FULL_DRUM_PITCH_CLASSES,
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True,
        roll_output=True),
    train_examples_path=None,
    eval_examples_path=None,
)

# Trio Models
CONFIG_MAP['cat-trio_16bar_big'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.MultiOutCategoricalLstmDecoder(
            output_depths=[
                90,  # melody
                90,  # bass
                512,  # drums
            ])),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=256,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[2048, 2048, 2048],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.TrioConverter(
        steps_per_quarter=4,
        slice_bars=16,
        gap_bars=2),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hiercat-trio_16bar_big'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalMultiOutLstmDecoder(
            core_decoders=[
                lstm_models.CategoricalLstmDecoder(),
                lstm_models.CategoricalLstmDecoder(),
                lstm_models.CategoricalLstmDecoder()],
            output_depths=[
                90,  # melody
                90,  # bass
                512,  # drums
            ])),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=256,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            hierarchical_output_sizes=[16],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.TrioConverter(
        steps_per_quarter=4,
        slice_bars=16,
        gap_bars=2),
    train_examples_path=None,
    eval_examples_path=None,
)

# 16-bar Melody Models
CONFIG_MAP['cat-mel_16bar_big'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[2048, 2048, 2048],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotMelodyConverter(
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=16,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hiercat-mel_16bar_big'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalMultiOutLstmDecoder(
            core_decoders=[lstm_models.CategoricalLstmDecoder()],
            output_depths=[90])),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            hierarchical_output_sizes=[16],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotMelodyConverter(
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=16,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)
