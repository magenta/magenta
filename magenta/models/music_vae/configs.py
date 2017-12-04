"""Cofnigurations for MusicVAE models."""
import collections

from tensorflow.contrib.training import HParams
from magenta.models.music_vae import data
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter',
     'note_sequence_converter', 'train_examples_path', 'eval_examples_path'])):

  def values(self):
    return self._asdict()


config_map = {}


def update_config(config, update_map):
  config_map = config.values()
  config_map.update(update_map)
  return Config(**config_map)


def merge_hparams(hp1, hp2):
  """Merge hp1 and hp2, preferring hp2 when conflicting."""
  hparams_map = hp1.values()
  hparams_map.update(hp2.values())
  return HParams(**hparams_map)

# Melody
config_map['cat-mel_2bar_small'] = Config(
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
        skip_polyphony=True,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path='',
    eval_examples_path='',
)

config_map['cat-mel_2bar_med'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[1024],
            dec_rnn_size=[1024, 1024, 1024],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=True,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path='',
    eval_examples_path='',
)

config_map['cat-mel_2bar_big'] = Config(
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
        skip_polyphony=True,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=2,
        steps_per_quarter=4),
    train_examples_path='',
    eval_examples_path='',
)

# Drums
config_map['cat-drums_2bar_small'] = Config(
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
    note_sequence_converter=data.OneHotDrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        binary_input=True),
    train_examples_path='',
    eval_examples_path='',
)

config_map['cat-drums_2bar_med'] = Config(
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
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotDrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        binary_input=True),
    train_examples_path='',
    eval_examples_path='',
)

config_map['cat-drums_2bar_big'] = Config(
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
    note_sequence_converter=data.OneHotDrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        binary_input=True),
    train_examples_path='',
    eval_examples_path='',
)

# Trio Models
config_map['cat-trio_b16_h16_big'] = Config(
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
    train_examples_path='',
    eval_examples_path='',
)

config_map['hiercat-trio_b16_h16_big'] = Config(
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
    train_examples_path='',
    eval_examples_path='',
)

config_map['hiercat-trio_b16_h16_med'] = Config(
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
            enc_rnn_size=[1024, 1024],
            dec_rnn_size=[1024],
            hierarchical_output_sizes=[16],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.TrioConverter(
        steps_per_quarter=4,
        slice_bars=16,
        gap_bars=2),
    train_examples_path='',
    eval_examples_path='',
)

# 16-bar Models
config_map['cat-mel_16bar_med'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[1024, 1024],
            dec_rnn_size=[1024, 1024, 1024],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotMelodyConverter(
        valid_programs=data.MEL_PROGRAMS,
        skip_polyphony=True,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=16,
        steps_per_quarter=4),
    train_examples_path='',
    eval_examples_path='',
)

config_map['hiercat-mel_16bar_med'] = Config(
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
            enc_rnn_size=[1024, 1024],
            dec_rnn_size=[1024],
            hierarchical_output_sizes=[16],
        )),
    note_sequence_augmenter=None,
    note_sequence_converter=data.OneHotMelodyConverter(
        skip_polyphony=True,
        max_bars=100,  # Truncate long melodies before slicing.
        slice_bars=16,
        steps_per_quarter=4),
    train_examples_path='',
    eval_examples_path='',
)


