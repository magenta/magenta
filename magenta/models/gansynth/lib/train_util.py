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
"""Train a progressive GAN model.

See https://arxiv.org/abs/1710.10196 for details about the model.

See https://github.com/tkarras/progressive_growing_of_gans for the original
theano implementation.
"""

import os
import time

from absl import logging
from magenta.models.gansynth.lib import networks
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan


def make_train_sub_dir(stage_id, **kwargs):
  """Returns the log directory for training stage `stage_id`."""
  return os.path.join(kwargs['train_root_dir'], 'stage_{:05d}'.format(stage_id))


def make_resolution_schedule(**kwargs):
  """Returns an object of `ResolutionSchedule`."""
  return networks.ResolutionSchedule(
      scale_mode=kwargs['scale_mode'],
      start_resolutions=(kwargs['start_height'], kwargs['start_width']),
      scale_base=kwargs['scale_base'],
      num_resolutions=kwargs['num_resolutions'])


def get_stage_ids(**kwargs):
  """Returns a list of stage ids.

  Args:
    **kwargs: A dictionary of
        'train_root_dir': A string of root directory of training logs.
        'num_resolutions': An integer of number of progressive resolutions.
  """
  train_sub_dirs = [
      sub_dir for sub_dir in tf.gfile.ListDirectory(kwargs['train_root_dir'])
      if sub_dir.startswith('stage_')
  ]

  # If fresh start, start with start_stage_id = 0
  # If has been trained for n = len(train_sub_dirs) stages, start with the last
  # stage, i.e. start_stage_id = n - 1.
  start_stage_id = max(0, len(train_sub_dirs) - 1)

  return list(range(start_stage_id, get_total_num_stages(**kwargs)))


def get_total_num_stages(**kwargs):
  """Returns total number of training stages."""
  return 2 * kwargs['num_resolutions'] - 1


def get_batch_size(stage_id, **kwargs):
  """Returns batch size for each stage.

  It is expected that `len(batch_size_schedule) == num_resolutions`. Each stage
  corresponds to a resolution and hence a batch size. However if
  `len(batch_size_schedule) < num_resolutions`, pad `batch_size_schedule` in the
  beginning with the first batch size.

  Args:
    stage_id: An integer of training stage index.
    **kwargs: A dictionary of
        'batch_size_schedule': A list of integer, each element is the batch size
            for the current training image resolution.
        'num_resolutions': An integer of number of progressive resolutions.

  Returns:
    An integer batch size for the `stage_id`.
  """
  batch_size_schedule = kwargs['batch_size_schedule']
  num_resolutions = kwargs['num_resolutions']
  if len(batch_size_schedule) < num_resolutions:
    batch_size_schedule = (
        [batch_size_schedule[0]] * (num_resolutions - len(batch_size_schedule))
        + batch_size_schedule)

  return int(batch_size_schedule[(stage_id + 1) // 2])


def get_stage_info(stage_id, **kwargs):
  """Returns information for a training stage.

  Args:
    stage_id: An integer of training stage index.
    **kwargs: A dictionary of
        'num_resolutions': An integer of number of progressive resolutions.
        'stable_stage_num_images': An integer of number of training images in
            the stable stage.
        'transition_stage_num_images': An integer of number of training images
            in the transition stage.
        'total_num_images': An integer of total number of training images.

  Returns:
    A tuple of integers. The first entry is the number of blocks. The second
    entry is the accumulated total number of training images when stage
    `stage_id` is finished.

  Raises:
    ValueError: If `stage_id` is not in [0, total number of stages).
  """
  total_num_stages = get_total_num_stages(**kwargs)
  valid_stage_id = (0 <= stage_id < total_num_stages)
  if not valid_stage_id:
    raise ValueError(
        '`stage_id` must be in [0, {0}), but instead was {1}'.format(
            total_num_stages, stage_id))

  # Even stage_id: stable training stage.
  # Odd stage_id: transition training stage.
  num_blocks = (stage_id + 1) // 2 + 1
  num_images = ((stage_id // 2 + 1) * kwargs['stable_stage_num_images'] + (
      (stage_id + 1) // 2) * kwargs['transition_stage_num_images'])

  total_num_images = kwargs['total_num_images']
  if stage_id >= total_num_stages - 1:
    num_images = total_num_images
  num_images = min(num_images, total_num_images)

  return num_blocks, num_images


def make_latent_vectors(num, **kwargs):
  """Returns a batch of `num` random latent vectors."""
  return tf.random_normal([num, kwargs['latent_vector_size']], dtype=tf.float32)


def make_interpolated_latent_vectors(num_rows, num_columns, **kwargs):
  """Returns a batch of linearly interpolated latent vectors.

  Given two randomly generated latent vector za and zb, it can generate
  a row of `num_columns` interpolated latent vectors, i.e.
  [..., za + (zb - za) * i / (num_columns - 1), ...] where
  i = 0, 1, ..., `num_columns` - 1.

  This function produces `num_rows` such rows and returns a (flattened)
  batch of latent vectors with batch size `num_rows * num_columns`.

  Args:
    num_rows: An integer. Number of rows of interpolated latent vectors.
    num_columns: An integer. Number of interpolated latent vectors in each row.
    **kwargs: A dictionary of
        'latent_vector_size': An integer of latent vector size.

  Returns:
    A `Tensor` of shape `[num_rows * num_columns, latent_vector_size]`.
  """
  ans = []
  for _ in range(num_rows):
    z = tf.random_normal([2, kwargs['latent_vector_size']])
    r = tf.reshape(
        tf.to_float(tf.range(num_columns)) / (num_columns - 1), [-1, 1])
    dz = z[1] - z[0]
    ans.append(z[0] + tf.stack([dz] * num_columns) * r)
  return tf.concat(ans, axis=0)


def define_loss(gan_model, **kwargs):
  """Defines progressive GAN losses.

  The generator and discriminator both use wasserstein loss. In addition,
  a small penalty term is added to the discriminator loss to prevent it getting
  too large.

  Args:
    gan_model: A `GANModel` namedtuple.
    **kwargs: A dictionary of
        'gradient_penalty_weight': A float of gradient norm target for
            wasserstein loss.
        'gradient_penalty_target': A float of gradient penalty weight for
            wasserstein loss.
        'real_score_penalty_weight': A float of Additional penalty to keep
            the scores from drifting too far from zero.

  Returns:
    A `GANLoss` namedtuple.
  """
  gan_loss = tfgan.gan_loss(
      gan_model,
      generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
      discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
      gradient_penalty_weight=kwargs['gradient_penalty_weight'],
      gradient_penalty_target=kwargs['gradient_penalty_target'],
      gradient_penalty_epsilon=0.0)

  real_score_penalty = tf.reduce_mean(
      tf.square(gan_model.discriminator_real_outputs))
  tf.summary.scalar('real_score_penalty', real_score_penalty)

  return gan_loss._replace(
      discriminator_loss=(
          gan_loss.discriminator_loss +
          kwargs['real_score_penalty_weight'] * real_score_penalty))


def define_train_ops(gan_model, gan_loss, **kwargs):
  """Defines progressive GAN train ops.

  Args:
    gan_model: A `GANModel` namedtuple.
    gan_loss: A `GANLoss` namedtuple.
    **kwargs: A dictionary of
        'adam_beta1': A float of Adam optimizer beta1.
        'adam_beta2': A float of Adam optimizer beta2.
        'generator_learning_rate': A float of generator learning rate.
        'discriminator_learning_rate': A float of discriminator learning rate.

  Returns:
    A tuple of `GANTrainOps` namedtuple and a list variables tracking the state
    of optimizers.
  """
  with tf.variable_scope('progressive_gan_train_ops') as var_scope:
    beta1, beta2 = kwargs['adam_beta1'], kwargs['adam_beta2']
    gen_opt = tf.train.AdamOptimizer(kwargs['generator_learning_rate'], beta1,
                                     beta2)
    dis_opt = tf.train.AdamOptimizer(kwargs['discriminator_learning_rate'],
                                     beta1, beta2)
    gan_train_ops = tfgan.gan_train_ops(gan_model, gan_loss, gen_opt, dis_opt)
  return gan_train_ops, tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope.name)


def add_generator_smoothing_ops(generator_ema, gan_model, gan_train_ops):
  """Adds generator smoothing ops."""
  with tf.control_dependencies([gan_train_ops.generator_train_op]):
    new_generator_train_op = generator_ema.apply(gan_model.generator_variables)

  gan_train_ops = gan_train_ops._replace(
      generator_train_op=new_generator_train_op)
  generator_vars_to_restore = generator_ema.variables_to_restore(
      gan_model.generator_variables)
  return gan_train_ops, generator_vars_to_restore


def make_var_scope_custom_getter_for_ema(ema):
  """Makes variable scope custom getter."""

  def _custom_getter(getter, name, *args, **kwargs):
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var

  return _custom_getter


def add_model_summaries(model, **kwargs):
  """Adds model summaries.

  This function adds several useful summaries during training:
  - fake_images: A grid of fake images based on random latent vectors.
  - interp_images: A grid of fake images based on interpolated latent vectors.
  - real_images_blend: A grid of real images.
  - summaries for `gan_model` losses, variable distributions etc.

  Args:
    model: An model object having all information of progressive GAN model,
        e.g. the return of build_model().
    **kwargs: A dictionary of
      'fake_grid_size': The fake image grid size for summaries.
      'interp_grid_size': The latent space interpolated image grid size for
          summaries.
      'colors': Number of image channels.
      'latent_vector_size': An integer of latent vector size.
  """
  fake_grid_size = kwargs['fake_grid_size']
  interp_grid_size = kwargs['interp_grid_size']
  colors = kwargs['colors']

  image_shape = list(model.resolution_schedule.final_resolutions)

  fake_batch_size = fake_grid_size**2
  fake_images_shape = [fake_batch_size] + image_shape + [colors]

  interp_batch_size = interp_grid_size**2
  interp_images_shape = [interp_batch_size] + image_shape + [colors]

  # When making prediction, use the ema smoothed generator vars.
  with tf.variable_scope(
      model.gan_model.generator_scope,
      reuse=True,
      custom_getter=make_var_scope_custom_getter_for_ema(model.generator_ema)):
    z_fake = make_latent_vectors(fake_batch_size, **kwargs)
    fake_images = model.gan_model.generator_fn(z_fake)
    fake_images.set_shape(fake_images_shape)

    z_interp = make_interpolated_latent_vectors(interp_grid_size,
                                                interp_grid_size, **kwargs)
    interp_images = model.gan_model.generator_fn(z_interp)
    interp_images.set_shape(interp_images_shape)

  tf.summary.image(
      'fake_images',
      tfgan.eval.eval_utils.image_grid(
          fake_images,
          grid_shape=[fake_grid_size] * 2,
          image_shape=image_shape,
          num_channels=colors),
      max_outputs=1)

  tf.summary.image(
      'interp_images',
      tfgan.eval.eval_utils.image_grid(
          interp_images,
          grid_shape=[interp_grid_size] * 2,
          image_shape=image_shape,
          num_channels=colors),
      max_outputs=1)

  real_grid_size = int(np.sqrt(model.batch_size))
  tf.summary.image(
      'real_images_blend',
      tfgan.eval.eval_utils.image_grid(
          model.gan_model.real_data[:real_grid_size**2],
          grid_shape=(real_grid_size, real_grid_size),
          image_shape=image_shape,
          num_channels=colors),
      max_outputs=1)

  tfgan.eval.add_gan_model_summaries(model.gan_model)


def make_scaffold(stage_id, optimizer_var_list, **kwargs):
  """Makes a custom scaffold.

  The scaffold
  - restores variables from the last training stage.
  - initializes new variables in the new block.

  Args:
    stage_id: An integer of stage id.
    optimizer_var_list: A list of optimizer variables.
    **kwargs: A dictionary of
        'train_root_dir': A string of root directory of training logs.
        'num_resolutions': An integer of number of progressive resolutions.
        'stable_stage_num_images': An integer of number of training images in
            the stable stage.
        'transition_stage_num_images': An integer of number of training images
            in the transition stage.
        'total_num_images': An integer of total number of training images.

  Returns:
    A `Scaffold` object.
  """
  # Holds variables that from the previous stage and need to be restored.
  restore_var_list = []
  prev_ckpt = None
  curr_ckpt = tf.train.latest_checkpoint(make_train_sub_dir(stage_id, **kwargs))
  if stage_id > 0 and curr_ckpt is None:
    prev_ckpt = tf.train.latest_checkpoint(
        make_train_sub_dir(stage_id - 1, **kwargs))

    num_blocks, _ = get_stage_info(stage_id, **kwargs)
    prev_num_blocks, _ = get_stage_info(stage_id - 1, **kwargs)

    # Holds variables created in the new block of the current stage. If the
    # current stage is a stable stage (except the initial stage), this list
    # will be empty.
    new_block_var_list = []
    for block_id in range(prev_num_blocks + 1, num_blocks + 1):
      new_block_var_list.extend(
          tf.get_collection(
              tf.GraphKeys.GLOBAL_VARIABLES,
              scope='.*/{}/'.format(networks.block_name(block_id))))

    # Every variables that are 1) not for optimizers and 2) from the new block
    # need to be restored.
    restore_var_list = [
        var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        if var not in set(optimizer_var_list + new_block_var_list)
    ]

  # Add saver op to graph. This saver is used to restore variables from the
  # previous stage.
  saver_for_restore = tf.train.Saver(
      var_list=restore_var_list, allow_empty=True)
  # Add the op to graph that initializes all global variables.
  init_op = tf.global_variables_initializer()

  def _init_fn(unused_scaffold, sess):
    # First initialize every variables.
    sess.run(init_op)
    logging.info('\n'.join([var.name for var in restore_var_list]))
    # Then overwrite variables saved in previous stage.
    if prev_ckpt is not None:
      saver_for_restore.restore(sess, prev_ckpt)

  # Use a dummy init_op here as all initialization is done in init_fn.
  return tf.train.Scaffold(init_op=tf.constant([]), init_fn=_init_fn)


def make_status_message(model):
  """Makes a string `Tensor` of training status."""
  return tf.string_join(
      [
          'Starting train step: current_image_id: ',
          tf.as_string(model.current_image_id), ', progress: ',
          tf.as_string(model.progress), ', num_blocks: {}'.format(
              model.num_blocks), ', batch_size: {}'.format(model.batch_size)
      ],
      name='status_message')


class ProganDebugHook(tf.train.SessionRunHook):
  """Prints summary statistics of all tf variables."""

  def __init__(self):
    super(ProganDebugHook, self).__init__()
    self._fetches = list(tf.global_variables())

  def before_run(self, _):
    return tf.train.SessionRunArgs(self._fetches)

  def after_run(self, _, vals):
    print('=============')
    print('Weight stats:')
    for v, r in zip(self._fetches, vals.results):
      print('\t', v.name, np.min(r), np.mean(r), np.max(r), r.shape)
    print('=============')


class TrainTimeHook(tf.train.SessionRunHook):
  """Updates the train_time variable.

  Optionally stops training if we've passed a time limit.
  """
  _last_run_start_time = Ellipsis  # type: float

  def __init__(self, train_time, time_limit=None):
    super(TrainTimeHook, self).__init__()
    self._train_time = train_time
    self._time_limit = time_limit
    self._increment_amount = tf.placeholder(tf.float32, None)
    self._increment_op = tf.assign_add(train_time, self._increment_amount)
    self._last_run_duration = None

  def before_run(self, _):
    self._last_run_start_time = time.time()
    if self._last_run_duration is not None:
      return tf.train.SessionRunArgs(
          [self._train_time, self._increment_op],
          feed_dict={self._increment_amount: self._last_run_duration})
    else:
      return tf.train.SessionRunArgs([self._train_time])

  def after_run(self, run_context, vals):
    self._last_run_duration = time.time() - self._last_run_start_time
    train_time = vals.results[0]
    if (self._time_limit is not None) and (train_time > self._time_limit):
      run_context.request_stop()


def train(model, **kwargs):
  """Trains progressive GAN for stage `stage_id`.

  Args:
    model: An model object having all information of progressive GAN model,
        e.g. the return of build_model().
    **kwargs: A dictionary of
        'train_root_dir': A string of root directory of training logs.
        'master': Name of the TensorFlow master to use.
        'task': The Task ID. This value is used when training with multiple
            workers to identify each worker.
        'save_summaries_num_images': Save summaries in this number of images.
        'debug_hook': Whether to attach the debug hook to the training session.
  Returns:
    None.
  """
  logging.info('stage_id=%d, num_blocks=%d, num_images=%d', model.stage_id,
               model.num_blocks, model.num_images)

  scaffold = make_scaffold(model.stage_id, model.optimizer_var_list, **kwargs)

  logdir = make_train_sub_dir(model.stage_id, **kwargs)
  print('starting training, logdir: {}'.format(logdir))
  hooks = []
  if model.stage_train_time_limit is None:
    hooks.append(tf.train.StopAtStepHook(last_step=model.num_images))
  hooks.append(tf.train.LoggingTensorHook(
      [make_status_message(model)], every_n_iter=1))
  hooks.append(TrainTimeHook(model.train_time, model.stage_train_time_limit))
  if kwargs['debug_hook']:
    hooks.append(ProganDebugHook())
  tfgan.gan_train(
      model.gan_train_ops,
      logdir=logdir,
      get_hooks_fn=tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, 1)),
      hooks=hooks,
      master=kwargs['master'],
      is_chief=(kwargs['task'] == 0),
      scaffold=scaffold,
      save_checkpoint_secs=600,
      save_summaries_steps=(kwargs['save_summaries_num_images']))
