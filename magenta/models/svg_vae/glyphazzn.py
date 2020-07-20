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

"""Defines the GlyphAzznProblem."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from magenta.models.svg_vae import svg_utils
import numpy as np
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

# Raw dataset paths (from datagen_beam.py)
# (Run t2t datagen on GlyphAzznProblem to convert these into a t2t dataset)
RAW_STAT_FILE = '/path/to/glyphazzn-internal-stats-00000-of-00001'
RAW_DATA_FILES = '/path/to/glyphazzn-internal-train*'
URL_SPLITS = 'third_party/py/magenta/models/svg_vae/glyphazzn_urls_split.txt'


class IdentityEncoder(object):

  def encode(self, inputs):
    return inputs

  def decode(self, inputs):
    return inputs


@registry.register_problem
class GlyphAzznProblem(problem.Problem):
  """Defines the GlyphAzznProblem class."""

  @property
  def dataset_splits(self):
    """Data splits to produce and number of shards for each."""
    # 10% evaluation data
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 90,
    }, {
        'split': problem.DatasetSplit.TEST,
        'shards': 10,
    }]

  @property
  def is_generate_per_split(self):
    # the data comes pre-split. so we should not shuffle and split it again.
    # this also means generate_samples will be called twice (one per split)
    return True

  @property
  def has_inputs(self):
    return True

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        'inputs': IdentityEncoder(),
        'targets': IdentityEncoder()
    }

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    # ignore any encoding since we don't need that
    return self.generate_samples(data_dir, tmp_dir, dataset_split)

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    del tmp_dir  # unused argument
    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    split_paths = [(split['split'], filepath_fns[split['split']](
        data_dir, split['shards'], shuffled=False))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(
            self.generate_encoded_samples(data_dir, tmp_dir, split), paths)
    else:
      generator_utils.generate_files(
          self.generate_encoded_samples(
              data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

    generator_utils.shuffle_dataset(all_paths)

  @property
  def categorical(self):
    # indicates we're using one-hot categories for command type.
    return True

  @property
  def feature_dim(self):
    return 10

  @property
  def num_classes(self):
    return 30

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of target svg commands."""
    del tmp_dir  # unused argument
    if not hasattr(self, 'splits'):
      tf.logging.info(
          'Loading binary_fp: train/test from {}'.format(URL_SPLITS))
      self.splits = {}
      for line in tf.gfile.Open(URL_SPLITS, 'r').read().split('\n'):
        if line:
          line = line.split(', ')
          self.splits[line[0]] = line[1]

    if not tf.gfile.Exists(data_dir):
      tf.gfile.MakeDirs(data_dir)

    if not tf.gfile.Exists(os.path.join(data_dir, 'mean.npz')):
      # FIRST, COPY THE MEAN/STDEV INTO DATA_DIR, in npz format
      for serialized_stats in tf.python_io.tf_record_iterator(RAW_STAT_FILE):
        stats = tf.train.Example()
        stats.ParseFromString(serialized_stats)
        mean = np.array(stats.features.feature['mean'].float_list.value)
        stdev = np.array(stats.features.feature['stddev'].float_list.value)
        # also want to set mean[:4] to zeros and stdev[:4] to ones, because
        # these are the class labels
        mean = np.concatenate((np.zeros([4]), mean[4:]), axis=0)
        stdev = np.concatenate((np.ones([4]), stdev[4:]), axis=0)
        # finally, save
        np.save(tf.gfile.Open(os.path.join(data_dir, 'mean.npz'), 'w'), mean)
        np.save(tf.gfile.Open(os.path.join(data_dir, 'stdev.npz'), 'w'), stdev)
        logging.info('Generated mean and stdev npzs')

    for raw_data_file in tf.gfile.Glob(RAW_DATA_FILES):
      for serialized_example in tf.python_io.tf_record_iterator(raw_data_file):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # determing whether this example belongs to a fontset in train or test
        this_bfp = str(
            example.features.feature['binary_fp'].bytes_list.value[0])
        if this_bfp not in self.splits:
          # randomly sample 10% to be test, the rest is train
          should_be_test = np.random.random() < 0.1
          self.splits[this_bfp] = 'test' if should_be_test else 'train'

        if self.splits[this_bfp] != dataset_split:
          continue

        yield {
            'targets_sln': np.array(
                example.features.feature['seq_len'].int64_list.value).astype(
                    np.int64).tolist(),
            'targets_cls': np.array(
                example.features.feature['class'].int64_list.value).astype(
                    np.int64).tolist(),
            'targets_rel': np.array(
                example.features.feature['sequence'].float_list.value).astype(
                    np.float32).tolist(),
            'targets_rnd': np.array(
                example.features.feature['rendered'].float_list.value).astype(
                    np.float32).tolist()
        }

  def example_reading_spec(self):
    data_fields = {'targets_rel': tf.FixedLenFeature([51*10], tf.float32),
                   'targets_rnd': tf.FixedLenFeature([64*64], tf.float32),
                   'targets_sln': tf.FixedLenFeature([1], tf.int64),
                   'targets_cls': tf.FixedLenFeature([1], tf.int64)}

    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_example(self, example, unused_mode, hparams):
    """Time series are flat on disk, we un-flatten them back here."""
    if not hasattr(self, 'mean_npz'):
      mean_filename = os.path.join(hparams.data_dir, 'mean.npz')
      stdev_filename = os.path.join(hparams.data_dir, 'stdev.npz')
      with tf.gfile.Open(mean_filename, 'r') as f:
        self.mean_npz = np.load(f)
      with tf.gfile.Open(stdev_filename, 'r') as f:
        self.stdev_npz = np.load(f)

    example['targets_cls'] = tf.reshape(example['targets_cls'], [1])
    example['targets_sln'] = tf.reshape(example['targets_sln'], [1])

    example['targets_rel'] = tf.reshape(example['targets_rel'], [51, 1, 10])
    # normalize (via gaussian)
    example['targets_rel'] = (example['targets_rel'] -
                              self.mean_npz) / self.stdev_npz

    # redefine shape inside model!
    example['targets_psr'] = tf.reshape(example['targets_rnd'],
                                        [1, 64 * 64]) / 255.
    del example['targets_rnd']

    if hparams.just_render:
      # training vae mode, use the last image (rendered icon) as input & output
      example['inputs'] = example['targets_psr'][-1, :]
      example['targets'] = example['targets_psr'][-1, :]
    else:
      example['inputs'] = tf.identity(example['targets_rel'])
      example['targets'] = tf.identity(example['targets_rel'])

    return example

  def hparams(self, defaults, model_hparams):
    p = defaults
    p.stop_at_eos = int(False)
    p.vocab_size = {'inputs': self.feature_dim, 'targets': self.feature_dim}
    p.modality = {'inputs': modalities.ModalityType.IDENTITY,
                  'targets': modalities.ModalityType.IDENTITY}

  @property
  def decode_hooks(self):
    to_img = svg_utils.create_image_conversion_fn(
        1, categorical=self.categorical)

    def sample_image(decode_hook_args):
      """Converts decoded predictions into summaries."""
      hparams = decode_hook_args.hparams

      if not hasattr(self, 'mean_npz'):
        mean_filename = os.path.join(hparams.data_dir, 'mean.npz')
        stdev_filename = os.path.join(hparams.data_dir, 'stdev.npz')
        with tf.gfile.open(mean_filename, 'r') as f:
          self.mean_npz = np.load(f)
        with tf.gfile.open(stdev_filename, 'r') as f:
          self.stdev_npz = np.load(f)

      values = []
      for pred_dict in decode_hook_args.predictions[0]:
        if hparams.just_render:
          # vae mode, outputs is image, just do image summary and continue
          values.append(svg_utils.make_image_summary(
              pred_dict['outputs'], 'rendered_outputs'))
          values.append(svg_utils.make_image_summary(
              pred_dict['targets'], 'rendered_targets'))
          continue

        if common_layers.shape_list(pred_dict['targets'])[0] == 1:
          continue

        # undo normalize (via gaussian)
        denorm_outputs = (pred_dict['outputs'] * self.stdev_npz) + self.mean_npz
        denorm_targets = (pred_dict['targets'] * self.stdev_npz) + self.mean_npz

        # simple cmds are 10 dim (4 one-hot, 6 args).
        # Convert to full SVG spec dimensionality so we can convert it to text.
        denorm_outputs = svg_utils.make_simple_cmds_long(denorm_outputs)
        denorm_targets = svg_utils.make_simple_cmds_long(denorm_targets)

        # sampled text summary
        output_svg = to_img([np.reshape(denorm_outputs, [-1, 30])])
        values.append(svg_utils.make_text_summary_value(output_svg,
                                                        'img/sampled'))

        # original text summary
        target_svg = to_img([np.reshape(denorm_targets, [-1, 30])])
        values.append(svg_utils.make_text_summary_value(target_svg, 'img/og'))

      return values
    return [sample_image]

  def eval_metrics(self):
    return []
