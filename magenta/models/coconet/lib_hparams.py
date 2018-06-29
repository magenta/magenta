"""Classes for defining hypermaters and model architectures."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools as it
import os
# internal imports
import numpy as np
import six
import tensorflow as tf
import yaml
from magenta.models.coconet import lib_util


class ModelMisspecificationError(Exception):
  """Exception for specifying a model that is not currently supported."""
  pass


def load_hparams(checkpoint_path):
  # hparams_fpath = os.path.join(os.path.dirname(checkpoint_path), 'config')
  hparams_fpath = os.path.join(checkpoint_path, 'config')
  with tf.gfile.Open(hparams_fpath, 'r') as p:
    hparams = Hyperparameters.load(p)
  return hparams


class Hyperparameters(object):
  """Stores hyperparameters for initialization, batch norm and training."""
  _defaults = dict(
      # Data.
      dataset=None,
      quantization_level=0.125,
      qpm=60,
      corrupt_ratio=0.25,
      # Input dimensions.
      batch_size=20,
      min_pitch=36,
      max_pitch=81,
      crop_piece_len=64,
      num_instruments=4,
      separate_instruments=True,
      # Batch norm parameters.
      batch_norm=True,
      batch_norm_variance_epsilon=1e-7,
      # Initialization.
      init_scale=0.1,
      # Model architecture.
      architecture='straight',
      # Hparams for depthwise separable convs.
      use_sep_conv=False,
      sep_conv_depth_multiplier=1,
      num_initial_regular_conv_layers=2,
      # Hparams for reducing pointwise in separable convs.
      num_pointwise_splits=1,
      # Hparams for dilated convs.
      num_dilation_blocks=3,
      dilate_time_only=False,
      repeat_last_dilation_level=False,
      # num_layers is used only for non dilated convs
      # as the number of layers in dilated convs is computed based on
      # num_dilation_blocks.
      num_layers=28,
      num_filters=256,
      use_residual=True,
      checkpoint_name=None,
      # Loss setup.
      # TODO(annahuang): currently maskout_method here is not functional,
      # still need to go through config_tools.
      maskout_method='orderless',
      optimize_mask_only=False,
      # use_softmax_loss=True,
      rescale_loss=True,
      # Training.
      # learning_rate=2**-6,
      learning_rate=2**-4,  # for sigmoids.
      mask_indicates_context=False,
      eval_freq=1,
      num_epochs=0,
      patience=5,
      # Runtime configs.
      run_dir=None,
      log_process=True,
      save_model_secs=30,
      run_id='')

  def __init__(self, *unused_args, **init_hparams):
    """Update the default parameters through string or keyword arguments.

    This __init__ provides two ways to initialize default parameters, either by
    passing a string representation of a a Python dictionary containing
    hyperparameter to value mapping or by passing those hyperparameter values
    directly as keyword arguments.

    Args:
      *unused_args: A tuple of arguments. This first expected argument is a
          string representation of a Python dictionary containing hyperparameter
          to value mapping. For example, {"num_layers":8, "num_filters"=128}.
      **init_hparams: Keyword arguments for setting hyperparameters.

    Raises:
      ValueError: When incoming hparams are not in class _defaults.
    """
    print('Instantiating hparams...')
    unknown_params = set(init_hparams) - set(Hyperparameters._defaults)
    if unknown_params:
      raise ValueError('Unknown hyperparameters: %s' % unknown_params)
    self.update(Hyperparameters._defaults)
    self.update(init_hparams)

  def update(self, dikt, **kwargs):
    for key, value in it.chain(six.iteritems(dikt), six.iteritems(kwargs)):
      setattr(self, key, value)

  @property
  def num_pitches(self):
    return self.max_pitch + 1 - self.min_pitch

  @property
  def input_depth(self):
    return self.num_instruments * 2

  @property
  def output_depth(self):
    return self.num_instruments if self.separate_instruments else 1

  @property
  def log_subdir_str(self):
    return '%s_%s' % (self.get_conv_arch().name, self.__str__())

  @property
  def name(self):
    return self.conv_arch.name

  @property
  def pianoroll_shape(self):
    if self.separate_instruments:
      return [self.crop_piece_len, self.num_pitches, self.num_instruments]
    else:
      return [self.crop_piece_len, self.num_pitches, 1]

  @property
  def use_softmax_loss(self):
    if not self.separate_instruments and (self.num_instruments > 1 or
                                          self.num_instruments == 0):
      return False
    else:
      return True

  def __str__(self):
    """Get all hyperparameters as a string."""
    # include whitelisted keys only
    shorthand = dict(
        batch_size='bs',
        learning_rate='lr',
        optimize_mask_only='mask_only',
        corrupt_ratio='corrupt',
        crop_piece_len='len',
        use_softmax_loss='soft',
        num_instruments='num_i',
        num_pitches='n_pch',
        quantization_level='quant',
        use_residual='res',
        use_sep_conv='sconv',
        sep_conv_depth_multiplier='depth_mul',
        num_initial_regular_conv_layers='nreg_conv',
        separate_instruments='sep',
        rescale_loss='rescale',
        maskout_method='mm')
    sorted_keys = sorted(shorthand.keys())
    line = ','.join(
        '%s=%s' % (shorthand[key], getattr(self, key)) for key in sorted_keys)
    return line

  def get_conv_arch(self):
    """Returns the model architecture."""
    return Architecture.make(
        self.architecture,
        self.input_depth,
        self.num_layers,
        self.num_filters,
        self.num_pitches,
        self.output_depth,
        crop_piece_len=self.crop_piece_len,
        num_dilation_blocks=self.num_dilation_blocks,
        dilate_time_only=self.dilate_time_only,
        repeat_last_dilation_level=self.repeat_last_dilation_level)

  def dump(self, file_object):
    yaml.dump(self.__dict__, file_object)

  @staticmethod
  def load(file_object):
    params_dict = yaml.load(file_object)
    hparams = Hyperparameters()
    hparams.update(params_dict)
    return hparams


class Architecture(lib_util.Factory):
  pass


class Straight(Architecture):
  """A convolutional net where each layer has the same number of filters."""
  key = 'straight'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches,
               output_depth, **kwargs):
    print('model_type=%s, input_depth=%d, output_depth=%d' %
          (self.key, input_depth, output_depth))
    assert num_layers >= 4

    self.layers = []

    def _add(**kwargs):
      self.layers.append(kwargs)

    _add(filters=[3, 3, input_depth, num_filters])
    for _ in range(num_layers - 3):
      _add(filters=[3, 3, num_filters, num_filters])
    _add(filters=[2, 2, num_filters, num_filters])
    _add(
        filters=[2, 2, num_filters, output_depth], activation=lib_util.identity)

    print('num_layers=%d, num_filters=%d' % (len(self.layers), num_filters))
    self.name = '%s-%d-%d' % (self.key, len(self.layers), num_filters)

  def __str__(self):
    return self.name


class Dilated(Architecture):
  """A dilated convnet where each layer has the same number of filters."""
  key = 'dilated'

  def __init__(self, input_depth, num_layers, num_filters, num_pitches,
               output_depth, **kwargs):
    tf.logging.info('model_type=%s, input_depth=%d, output_depth=%d' %
                    (self.key, input_depth, output_depth))
    kws = """num_dilation_blocks dilate_time_only crop_piece_len
          repeat_last_dilation_level"""
    for kw in kws.split():
      assert kw in kwargs
    num_dilation_blocks = kwargs['num_dilation_blocks']
    assert num_dilation_blocks >= 1
    dilate_time_only = kwargs['dilate_time_only']

    def compute_max_dilation_level(length):
      return int(np.ceil(np.log2(length))) - 1

    max_time_dilation_level = (
        compute_max_dilation_level(kwargs['crop_piece_len']))
    max_pitch_dilation_level = (
        compute_max_dilation_level(num_pitches))
    max_dilation_level = max(max_time_dilation_level, max_pitch_dilation_level)
    if kwargs['repeat_last_dilation_level']:
      tf.logging.info('Increasing max dilation level from %s to %s'
                      % (max_dilation_level, max_dilation_level + 1))
      max_dilation_level += 1

    def determine_dilation_rate(level, max_level):
      dilation_level = min(level, max_level)
      return 2 ** dilation_level

    self.layers = []

    def _add(**kwargs):
      self.layers.append(kwargs)

    _add(filters=[3, 3, input_depth, num_filters])
    for _ in range(num_dilation_blocks):
      for level in range(max_dilation_level + 1):
        time_dilation_rate = determine_dilation_rate(
            level, max_time_dilation_level)
        pitch_dilation_rate = determine_dilation_rate(
            level, max_pitch_dilation_level)
        if dilate_time_only:
          layer_dilation_rates = [time_dilation_rate, 1]
        else:
          layer_dilation_rates = [time_dilation_rate, pitch_dilation_rate]
        tf.logging.info('layer_dilation_rates %r' % layer_dilation_rates)
        _add(filters=[3, 3, num_filters, num_filters],
             dilation_rate=layer_dilation_rates)
    _add(filters=[2, 2, num_filters, num_filters])
    _add(
        filters=[2, 2, num_filters, output_depth], activation=lib_util.identity)

    tf.logging.info('num_layers=%d, num_filters=%d' % (
        len(self.layers), num_filters))
    self.name = '%s-%d-%d' % (self.key, len(self.layers), num_filters)

  def __str__(self):
    return self.name
