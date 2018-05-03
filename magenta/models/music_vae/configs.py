"""Configurations for MusicVAE models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from magenta.common import merge_hparams
from magenta.models.music_vae import data
from magenta.models.music_vae import data_hierarchical
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE
import magenta.music as mm

from tensorflow.contrib.training import HParams


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path'])):

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
            free_bits=0,
            max_beta=0.2,
            beta_rate=0.99999,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
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
            free_bits=0,
            max_beta=0.5,
            beta_rate=0.99999,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(transpose_range=(-5, 5)),
    data_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=False,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path=None,
    eval_examples_path=None,
)

# Chord-Conditioned Melody
CONFIG_MAP['cat-mel_2bar_med_chords'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[1024],
            dec_rnn_size=[512, 512, 512],
        )),
    note_sequence_augmenter=data.NoteSequenceAugmenter(
        transpose_range=(-3, 3)),
    data_converter=data.OneHotMelodyConverter(
        max_bars=100,
        slice_bars=2,
        steps_per_quarter=4,
        chord_encoding=mm.TriadChordOneHotEncoding()),
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
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
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
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
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
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
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
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
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
trio_16bar_converter = data.TrioConverter(
    steps_per_quarter=4,
    slice_bars=16,
    gap_bars=2)

CONFIG_MAP['flat-trio_16bar'] = Config(
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
    data_converter=trio_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hierdec-trio_16bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.SplitMultiOutLstmDecoder(
                core_decoders=[
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder()],
                output_depths=[
                    90,  # melody
                    90,  # bass
                    512,  # drums
                ]),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=256,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=trio_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hier-trio_16bar'] = Config(
    model=MusicVAE(
        lstm_models.HierarchicalLstmEncoder(
            lstm_models.BidirectionalLstmEncoder, [16, 16]),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.SplitMultiOutLstmDecoder(
                core_decoders=[
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder(),
                    lstm_models.CategoricalLstmDecoder()],
                output_depths=[
                    90,  # melody
                    90,  # bass
                    512,  # drums
                ]),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=256,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[1024],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=trio_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

# 16-bar Melody Models
mel_16bar_converter = data.OneHotMelodyConverter(
    skip_polyphony=False,
    max_bars=100,  # Truncate long melodies before slicing.
    slice_bars=16,
    steps_per_quarter=4)

CONFIG_MAP['flat-mel_16bar'] = Config(
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
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hierdec-mel_16bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hier-mel_16bar'] = Config(
    model=MusicVAE(
        lstm_models.HierarchicalLstmEncoder(
            lstm_models.BidirectionalLstmEncoder, [16, 16]),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[1024],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

# Multitrack
multiperf_encoder = lstm_models.HierarchicalLstmEncoder(
    lstm_models.BidirectionalLstmEncoder,
    level_lengths=[64, 8])
multiperf_decoder = lstm_models.HierarchicalLstmDecoder(
    lstm_models.CategoricalLstmDecoder(),
    level_lengths=[8, 64],
    disable_autoregression=True)

multiperf_hparams_med = merge_hparams(
    lstm_models.get_default_hparams(),
    HParams(
        batch_size=256,
        max_seq_len=512,
        z_size=512,
        enc_rnn_size=[1024],
        dec_rnn_size=[512, 512, 512]))

multiperf_hparams_big = merge_hparams(
    lstm_models.get_default_hparams(),
    HParams(
        batch_size=256,
        max_seq_len=512,
        z_size=512,
        enc_rnn_size=[2048],
        dec_rnn_size=[1024, 1024, 1024]))

CONFIG_MAP['hier-multiperf_vel_1bar_med'] = Config(
    model=MusicVAE(multiperf_encoder, multiperf_decoder),
    hparams=multiperf_hparams_med,
    note_sequence_augmenter=data.NoteSequenceAugmenter(
        transpose_range=(-3, 3)),
    data_converter=data_hierarchical.MultiInstrumentPerformanceConverter(
        num_velocity_bins=8,
        hop_size_bars=1,
        max_num_instruments=8,
        max_events_per_instrument=64,
    ),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hier-multiperf_vel_1bar_big'] = Config(
    model=MusicVAE(multiperf_encoder, multiperf_decoder),
    hparams=multiperf_hparams_big,
    note_sequence_augmenter=data.NoteSequenceAugmenter(
        transpose_range=(-3, 3)),
    data_converter=data_hierarchical.MultiInstrumentPerformanceConverter(
        num_velocity_bins=8,
        hop_size_bars=1,
        max_num_instruments=8,
        max_events_per_instrument=64,
    ),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hier-multiperf_vel_1bar_med_chords'] = Config(
    model=MusicVAE(multiperf_encoder, multiperf_decoder),
    hparams=multiperf_hparams_med,
    note_sequence_augmenter=data.NoteSequenceAugmenter(
        transpose_range=(-3, 3)),
    data_converter=data_hierarchical.MultiInstrumentPerformanceConverter(
        num_velocity_bins=8,
        hop_size_bars=1,
        max_num_instruments=8,
        max_events_per_instrument=64,
        chord_encoding=mm.TriadChordOneHotEncoding(),
    ),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['hier-multiperf_vel_1bar_big_chords'] = Config(
    model=MusicVAE(multiperf_encoder, multiperf_decoder),
    hparams=multiperf_hparams_big,
    note_sequence_augmenter=data.NoteSequenceAugmenter(
        transpose_range=(-3, 3)),
    data_converter=data_hierarchical.MultiInstrumentPerformanceConverter(
        num_velocity_bins=8,
        hop_size_bars=1,
        max_num_instruments=8,
        max_events_per_instrument=64,
        chord_encoding=mm.TriadChordOneHotEncoding(),
    ),
    train_examples_path=None,
    eval_examples_path=None,
)
