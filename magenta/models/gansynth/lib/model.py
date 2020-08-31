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
"""GANSynth Model class definition.

Exposes external API for generating samples and evaluation.
"""

import json
import os
import time

from magenta.models.gansynth.lib import data_helpers
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import network_functions as net_fns
from magenta.models.gansynth.lib import networks
from magenta.models.gansynth.lib import train_util
from magenta.models.gansynth.lib import util
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan


def set_flags(flags):
  """Set default hyperparameters."""
  # Must be specified externally
  flags.set_if_empty('train_root_dir', '/tmp/gansynth/train')
  flags.set_if_empty('train_data_path', '/tmp/gansynth/nsynth-train.tfrecord')

  ### Dataset ###
  flags.set_if_empty('dataset_name', 'nsynth_tfds')
  flags.set_if_empty('data_type', 'mel')  # linear, phase, mel
  flags.set_if_empty('audio_length', 64000)
  flags.set_if_empty('sample_rate', 16000)
  # specgrams_simple_normalizer, specgrams_freq_normalizer
  flags.set_if_empty('data_normalizer', 'specgrams_prespecified_normalizer')
  flags.set_if_empty('normalizer_margin', 0.8)
  flags.set_if_empty('normalizer_num_examples', 1000)
  flags.set_if_empty('mag_normalizer_a', 0.0661371661726)
  flags.set_if_empty('mag_normalizer_b', 0.113718730221)
  flags.set_if_empty('p_normalizer_a', 0.8)
  flags.set_if_empty('p_normalizer_b', 0.)

  ### Losses ###
  # Gradient norm target for wasserstein loss
  flags.set_if_empty('gradient_penalty_target', 1.0)
  flags.set_if_empty('gradient_penalty_weight', 10.0)
  # Additional penalty to keep the scores from drifting too far from zero
  flags.set_if_empty('real_score_penalty_weight', 0.001)
  # Auxiliary loss for conditioning
  flags.set_if_empty('generator_ac_loss_weight', 1.0)
  flags.set_if_empty('discriminator_ac_loss_weight', 1.0)
  # Weight of G&L consistency loss
  flags.set_if_empty('gen_gl_consistency_loss_weight', 0.0)

  ### Optimization ###
  flags.set_if_empty('generator_learning_rate', 0.0004)  # Learning rate
  flags.set_if_empty('discriminator_learning_rate', 0.0004)  # Learning rate
  flags.set_if_empty('adam_beta1', 0.0)  # Adam beta 1
  flags.set_if_empty('adam_beta2', 0.99)  # Adam beta 2
  flags.set_if_empty('fake_batch_size', 16)  # The fake image batch size

  ### Distributed Training ###
  flags.set_if_empty('master', '')  # Name of the TensorFlow master to use
  # The number of parameter servers. If the value is 0, then the parameters
  # are handled locally by the worker
  flags.set_if_empty('ps_tasks', 0)
  # The Task ID. This value is used when training with multiple workers to
  # identify each worker
  flags.set_if_empty('task', 0)

  ### Debugging ###
  flags.set_if_empty('debug_hook', False)

  ### -----------  HPARAM Settings for testing eval ###
  ### Progressive Training ###
  # A list of batch sizes for each resolution, if len(batch_size_schedule)
  # < num_resolutions, pad the schedule in the beginning with first batch size
  flags.set_if_empty('batch_size_schedule', [16, 8])
  flags.set_if_empty('stable_stage_num_images', 32)
  flags.set_if_empty('transition_stage_num_images', 32)
  flags.set_if_empty('total_num_images', 320)
  flags.set_if_empty('save_summaries_num_images', 100)
  flags.set_if_empty('train_progressive', True)
  # For fixed-wall-clock-time training, total training time limit in sec.
  # If specified, overrides the iteration limits.
  flags.set_if_empty('train_time_limit', None)
  flags.set_if_empty('train_time_stage_multiplier', 1.)

  ### Architecture ###
  # Choose an architecture function
  flags.set_if_empty('g_fn', 'specgram')  # 'specgram'
  flags.set_if_empty('d_fn', 'specgram')
  flags.set_if_empty('latent_vector_size', 256)
  flags.set_if_empty('kernel_size', 3)
  flags.set_if_empty('start_height', 4)  # Start specgram height
  flags.set_if_empty('start_width', 8)  # Start specgram width
  flags.set_if_empty('scale_mode', 'ALL')  # Scale mode ALL|H
  flags.set_if_empty('scale_base', 2)  # Resolution multiplier
  flags.set_if_empty('num_resolutions', 7)  # N progressive resolutions
  flags.set_if_empty('simple_arch', False)
  flags.set_if_empty('to_rgb_activation', 'tanh')
  # Number of filters
  flags.set_if_empty('fmap_base', 512)  # Base number of filters
  flags.set_if_empty('fmap_decay', 1.0)
  flags.set_if_empty('fmap_max', 128)


class Model(object):
  """Progressive GAN model for a specific stage and batch_size."""

  @classmethod
  def load_from_path(cls, path, flags=None):
    """Instantiate a Model for eval using flags and weights from a saved model.

    Currently only supports models trained by the experiment runner, since
    Model itself doesn't save flags (so we rely the runner's experiment.json)

    Args:
      path: Path to model directory (which contains stage folders).
      flags: Additional flags for loading the model.

    Raises:
      ValueError: If folder of path contains no stage folders.

    Returns:
      model: Instantiated model with saved weights.
    """
    # Read the flags from the experiment.json file
    # experiment.json is in the folder above
    # Remove last '/' if present
    path = path.rstrip('/')
    if not path.startswith('gs://'):
      path = util.expand_path(path)
    if flags is None:
      flags = lib_flags.Flags()
    flags['train_root_dir'] = path
    experiment_json_path = os.path.join(path, 'experiment.json')
    try:
      # Read json to dict
      with tf.gfile.GFile(experiment_json_path, 'r') as f:
        experiment_json = json.load(f)
      # Load dict as a Flags() object
      flags.load(experiment_json)
    except Exception as e:  # pylint: disable=broad-except
      print("Warning! Couldn't load model flags from experiment.json")
      print(e)
    # Set default flags
    set_flags(flags)
    flags.print_values()
    # Get list_of_directories
    train_sub_dirs = sorted([sub_dir for sub_dir in tf.gfile.ListDirectory(path)
                             if sub_dir.startswith('stage_')])
    if not train_sub_dirs:
      raise ValueError('No stage folders found, is %s the correct model path?'
                       % path)
    # Get last checkpoint
    last_stage_dir = train_sub_dirs[-1]
    stage_id = int(last_stage_dir.split('_')[-1].strip('/'))
    weights_dir = os.path.join(path, last_stage_dir)
    ckpt = tf.train.latest_checkpoint(weights_dir)
    print('Load model from {}'.format(ckpt))
    # Load the model, use eval_batch_size if present
    batch_size = flags.get('eval_batch_size',
                           train_util.get_batch_size(stage_id, **flags))
    model = cls(stage_id, batch_size, flags)
    model.saver.restore(model.sess, ckpt)
    return model

  def __init__(self, stage_id, batch_size, config):
    """Build graph stage from config dictionary.

    Stage_id and batch_size change during training so they are kept separate
    from the global config. This function is also called by 'load_from_path()'.

    Args:
      stage_id: (int) Build generator/discriminator with this many stages.
      batch_size: (int) Build graph with fixed batch size.
      config: (dict) All the global state.
    """
    data_helper = data_helpers.registry[config['data_type']](config)
    real_images, real_one_hot_labels = data_helper.provide_data(batch_size)

    # gen_one_hot_labels = real_one_hot_labels
    gen_one_hot_labels = data_helper.provide_one_hot_labels(batch_size)
    num_tokens = int(real_one_hot_labels.shape[1])

    current_image_id = tf.train.get_or_create_global_step()
    current_image_id_inc_op = current_image_id.assign_add(batch_size)
    tf.summary.scalar('current_image_id', current_image_id)

    train_time = tf.Variable(0., dtype=tf.float32, trainable=False)
    tf.summary.scalar('train_time', train_time)

    resolution_schedule = train_util.make_resolution_schedule(**config)
    num_blocks, num_images = train_util.get_stage_info(stage_id, **config)

    num_stages = (2*config['num_resolutions']) - 1
    if config['train_time_limit'] is not None:
      stage_times = np.zeros(num_stages, dtype='float32')
      stage_times[0] = 1.
      for i in range(1, num_stages):
        stage_times[i] = (stage_times[i-1] *
                          config['train_time_stage_multiplier'])
      stage_times *= config['train_time_limit'] / np.sum(stage_times)
      stage_times = np.cumsum(stage_times)
      print('Stage times:')
      for t in stage_times:
        print('\t{}'.format(t))

    if config['train_progressive']:
      if config['train_time_limit'] is not None:
        progress = networks.compute_progress_from_time(
            train_time, config['num_resolutions'], num_blocks, stage_times)
      else:
        progress = networks.compute_progress(
            current_image_id, config['stable_stage_num_images'],
            config['transition_stage_num_images'], num_blocks)
    else:
      progress = num_blocks - 1.  # Maximum value, must be float.
      num_images = 0
      for stage_id_idx in train_util.get_stage_ids(**config):
        _, n = train_util.get_stage_info(stage_id_idx, **config)
        num_images += n

    # Add to config
    config['resolution_schedule'] = resolution_schedule
    config['num_blocks'] = num_blocks
    config['num_images'] = num_images
    config['progress'] = progress
    config['num_tokens'] = num_tokens
    tf.summary.scalar('progress', progress)

    real_images = networks.blend_images(
        real_images, progress, resolution_schedule, num_blocks=num_blocks)

    ########## Define model.
    noises = train_util.make_latent_vectors(batch_size, **config)

    # Get network functions and wrap with hparams
    # pylint:disable=unnecessary-lambda
    g_fn = lambda x: net_fns.g_fn_registry[config['g_fn']](x, **config)
    d_fn = lambda x: net_fns.d_fn_registry[config['d_fn']](x, **config)
    # pylint:enable=unnecessary-lambda

    # Extra lambda functions to conform to tfgan.gan_model interface
    gan_model = tfgan.gan_model(
        generator_fn=lambda inputs: g_fn(inputs)[0],
        discriminator_fn=lambda images, unused_cond: d_fn(images)[0],
        real_data=real_images,
        generator_inputs=(noises, gen_one_hot_labels))

    ########## Define loss.
    gan_loss = train_util.define_loss(gan_model, **config)

    ########## Auxiliary loss functions
    def _compute_ac_loss(images, target_one_hot_labels):
      with tf.variable_scope(gan_model.discriminator_scope, reuse=True):
        _, end_points = d_fn(images)
      return tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits_v2(
              labels=tf.stop_gradient(target_one_hot_labels),
              logits=end_points['classification_logits']))

    def _compute_gl_consistency_loss(data):
      """G&L consistency loss."""
      sh = data_helper.specgrams_helper
      is_mel = isinstance(data_helper, data_helpers.DataMelHelper)
      if is_mel:
        stfts = sh.melspecgrams_to_stfts(data)
      else:
        stfts = sh.specgrams_to_stfts(data)
      waves = sh.stfts_to_waves(stfts)
      new_stfts = sh.waves_to_stfts(waves)
      # Magnitude loss
      mag = tf.abs(stfts)
      new_mag = tf.abs(new_stfts)
      # Normalize loss to max
      get_max = lambda x: tf.reduce_max(x, axis=(1, 2), keepdims=True)
      mag_max = get_max(mag)
      new_mag_max = get_max(new_mag)
      mag_scale = tf.maximum(1.0, tf.maximum(mag_max, new_mag_max))
      mag_diff = (mag - new_mag) / mag_scale
      mag_loss = tf.reduce_mean(tf.square(mag_diff))
      return mag_loss

    with tf.name_scope('losses'):
      # Loss weights
      gen_ac_loss_weight = config['generator_ac_loss_weight']
      dis_ac_loss_weight = config['discriminator_ac_loss_weight']
      gen_gl_consistency_loss_weight = config['gen_gl_consistency_loss_weight']

      # AC losses.
      fake_ac_loss = _compute_ac_loss(gan_model.generated_data,
                                      gen_one_hot_labels)
      real_ac_loss = _compute_ac_loss(gan_model.real_data, real_one_hot_labels)

      # GL losses.
      is_fourier = isinstance(data_helper, (data_helpers.DataSTFTHelper,
                                            data_helpers.DataSTFTNoIFreqHelper,
                                            data_helpers.DataMelHelper))
      if isinstance(data_helper, data_helpers.DataWaveHelper):
        is_fourier = False

      if is_fourier:
        fake_gl_loss = _compute_gl_consistency_loss(gan_model.generated_data)
        real_gl_loss = _compute_gl_consistency_loss(gan_model.real_data)

      # Total losses.
      wx_fake_ac_loss = gen_ac_loss_weight * fake_ac_loss
      wx_real_ac_loss = dis_ac_loss_weight * real_ac_loss
      wx_fake_gl_loss = 0.0
      if (is_fourier and
          gen_gl_consistency_loss_weight > 0 and
          stage_id == train_util.get_total_num_stages(**config) - 1):
        wx_fake_gl_loss = fake_gl_loss * gen_gl_consistency_loss_weight
      # Update the loss functions
      gan_loss = gan_loss._replace(
          generator_loss=(
              gan_loss.generator_loss + wx_fake_ac_loss + wx_fake_gl_loss),
          discriminator_loss=(gan_loss.discriminator_loss + wx_real_ac_loss))

      tf.summary.scalar('fake_ac_loss', fake_ac_loss)
      tf.summary.scalar('real_ac_loss', real_ac_loss)
      tf.summary.scalar('wx_fake_ac_loss', wx_fake_ac_loss)
      tf.summary.scalar('wx_real_ac_loss', wx_real_ac_loss)
      tf.summary.scalar('total_gen_loss', gan_loss.generator_loss)
      tf.summary.scalar('total_dis_loss', gan_loss.discriminator_loss)

      if is_fourier:
        tf.summary.scalar('fake_gl_loss', fake_gl_loss)
        tf.summary.scalar('real_gl_loss', real_gl_loss)
        tf.summary.scalar('wx_fake_gl_loss', wx_fake_gl_loss)

    ########## Define train ops.
    gan_train_ops, optimizer_var_list = train_util.define_train_ops(
        gan_model, gan_loss, **config)
    gan_train_ops = gan_train_ops._replace(
        global_step_inc_op=current_image_id_inc_op)

    ########## Generator smoothing.
    generator_ema = tf.train.ExponentialMovingAverage(decay=0.999)
    gan_train_ops, generator_vars_to_restore = \
        train_util.add_generator_smoothing_ops(generator_ema,
                                               gan_model,
                                               gan_train_ops)
    load_scope = tf.variable_scope(
        gan_model.generator_scope,
        reuse=True,
        custom_getter=train_util.make_var_scope_custom_getter_for_ema(
            generator_ema))

    ########## Separate path for generating samples with a placeholder (ph)
    # Mapping of pitches to one-hot labels
    pitch_counts = data_helper.get_pitch_counts()
    pitch_to_label_dict = {}
    for i, pitch in enumerate(sorted(pitch_counts.keys())):
      pitch_to_label_dict[pitch] = i

    # (label_ph, noise_ph) -> fake_wave_ph
    labels_ph = tf.placeholder(tf.int32, [batch_size])
    noises_ph = tf.placeholder(tf.float32, [batch_size,
                                            config['latent_vector_size']])
    num_pitches = len(pitch_counts)
    one_hot_labels_ph = tf.one_hot(labels_ph, num_pitches)
    with load_scope:
      fake_data_ph, _ = g_fn((noises_ph, one_hot_labels_ph))
      fake_waves_ph = data_helper.data_to_waves(fake_data_ph)

    if config['train_time_limit'] is not None:
      stage_train_time_limit = stage_times[stage_id]
      #  config['train_time_limit'] * \
      # (float(stage_id+1) / ((2*config['num_resolutions'])-1))
    else:
      stage_train_time_limit = None

    ########## Add variables as properties
    self.stage_id = stage_id
    self.batch_size = batch_size
    self.config = config
    self.data_helper = data_helper
    self.resolution_schedule = resolution_schedule
    self.num_images = num_images
    self.num_blocks = num_blocks
    self.current_image_id = current_image_id
    self.progress = progress
    self.generator_fn = g_fn
    self.discriminator_fn = d_fn
    self.gan_model = gan_model
    self.fake_ac_loss = fake_ac_loss
    self.real_ac_loss = real_ac_loss
    self.gan_loss = gan_loss
    self.gan_train_ops = gan_train_ops
    self.optimizer_var_list = optimizer_var_list
    self.generator_ema = generator_ema
    self.generator_vars_to_restore = generator_vars_to_restore
    self.real_images = real_images
    self.real_one_hot_labels = real_one_hot_labels
    self.load_scope = load_scope
    self.pitch_counts = pitch_counts
    self.pitch_to_label_dict = pitch_to_label_dict
    self.labels_ph = labels_ph
    self.noises_ph = noises_ph
    self.fake_waves_ph = fake_waves_ph
    self.saver = tf.train.Saver()
    self.sess = tf.Session()
    self.train_time = train_time
    self.stage_train_time_limit = stage_train_time_limit

  def add_summaries(self):
    """Adds model summaries."""
    config = self.config
    data_helper = self.data_helper

    def _add_waves_summary(name, waves, max_outputs):
      tf.summary.audio(name,
                       waves,
                       sample_rate=config['sample_rate'],
                       max_outputs=max_outputs)

    def _add_specgrams_summary(name, specgrams, max_outputs):
      tf.summary.image(
          name + '_m', specgrams[:, :, :, 0:1], max_outputs=max_outputs)
      tf.summary.image(
          name + '_p', specgrams[:, :, :, 1:2], max_outputs=max_outputs)

    fake_batch_size = config['fake_batch_size']
    real_batch_size = self.batch_size
    real_one_hot_labels = self.real_one_hot_labels
    num_tokens = int(real_one_hot_labels.shape[1])

    # When making prediction, use the ema smoothed generator vars by
    # `_custom_getter`.
    with self.load_scope:
      noises = train_util.make_latent_vectors(fake_batch_size, **config)
      one_hot_labels = util.make_ordered_one_hot_vectors(fake_batch_size,
                                                         num_tokens)

      fake_images = self.gan_model.generator_fn((noises, one_hot_labels))
      real_images = self.real_images

    # Set shapes
    image_shape = list(self.resolution_schedule.final_resolutions)
    n_ch = 2
    fake_images.set_shape([fake_batch_size] + image_shape + [n_ch])
    real_images.set_shape([real_batch_size] + image_shape + [n_ch])

    # Generate waves and summaries
    # Convert to audio
    fake_waves = data_helper.data_to_waves(fake_images)
    real_waves = data_helper.data_to_waves(real_images)
    # Wave summaries
    _add_waves_summary('fake_waves', fake_waves, fake_batch_size)
    _add_waves_summary('real_waves', real_waves, real_batch_size)
    # Spectrogram summaries
    if isinstance(data_helper, data_helpers.DataWaveHelper):
      fake_images_spec = data_helper.specgrams_helper.waves_to_specgrams(
          fake_waves)
      real_images_spec = data_helper.specgrams_helper.waves_to_specgrams(
          real_waves)
      _add_specgrams_summary('fake_data', fake_images_spec, fake_batch_size)
      _add_specgrams_summary('real_data', real_images_spec, real_batch_size)
    else:
      _add_specgrams_summary('fake_data', fake_images, fake_batch_size)
      _add_specgrams_summary('real_data', real_images, real_batch_size)
    tfgan.eval.add_gan_model_summaries(self.gan_model)

  def _pitches_to_labels(self, pitches):
    return [self.pitch_to_label_dict[pitch] for pitch in pitches]

  def generate_z(self, n):
    return np.random.normal(size=[n, self.config['latent_vector_size']])

  def generate_samples(self, n_samples, pitch=None, max_audio_length=64000):
    """Generate random latent fake samples.

    If pitch is not specified, pitches will be sampled randomly.

    Args:
      n_samples: Number of samples to generate.
      pitch: List of pitches to generate.
      max_audio_length: Trim generation to <= this length.

    Returns:
      An array of audio for the notes [n_notes, n_audio_samples].
    """
    # Create list of pitches to generate
    if pitch is not None:
      pitches = [pitch]*n_samples
    else:
      all_pitches = []
      for k, v in self.pitch_counts.items():
        all_pitches.extend([k]*v)
      pitches = np.random.choice(all_pitches, n_samples)
    z = self.generate_z(len(pitches))
    return self.generate_samples_from_z(z, pitches, max_audio_length)

  def generate_samples_from_z(self, z, pitches, max_audio_length=64000):
    """Generate fake samples for given latents and pitches.

    Args:
      z: A numpy array of latent vectors [n_samples, n_latent dims].
      pitches: An iterable list of integer MIDI pitches [n_samples].
      max_audio_length: Integer, trim to this many samples.

    Returns:
      audio: Generated audio waveforms [n_samples, max_audio_length]
    """
    labels = self._pitches_to_labels(pitches)
    n_samples = len(labels)
    num_batches = int(np.ceil(float(n_samples) / self.batch_size))
    n_tot = num_batches * self.batch_size
    padding = n_tot - n_samples
    # Pads zeros to make batches even batch size.
    labels = labels + [0] * padding
    z = np.concatenate([z, np.zeros([padding, z.shape[1]])], axis=0)

    # Generate waves
    start_time = time.time()
    waves_list = []
    for i in range(num_batches):
      start = i * self.batch_size
      end = (i + 1) * self.batch_size

      waves = self.sess.run(self.fake_waves_ph,
                            feed_dict={self.labels_ph: labels[start:end],
                                       self.noises_ph: z[start:end]})
      # Trim waves
      for wave in waves:
        waves_list.append(wave[:max_audio_length, 0])

    # Remove waves corresponding to the padded zeros.
    result = np.stack(waves_list[:n_samples], axis=0)
    print('generate_samples: generated {} samples in {}s'.format(
        n_samples, time.time() - start_time))
    return result
