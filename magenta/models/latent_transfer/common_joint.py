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
"""Common functions/helpers for the joint model.

This library contains many comman functions and helpers used to train (using
script `train_joint.py`) the joint model (defined in `model_joint.py`). These
components are classified in the following categories:

  - Inetration helper that helps interate through data in the training loop.
    This includes:
        `BatchIndexIterator`, `InterGroupSamplingIndexIterator`,
        `GuasssianDataHelper`, `SingleDataIterator`, `PairedDataIterator`.

  - Summary helper that makes manual sumamrization easiser. This includes:
        `ManualSummaryHelper`.

  - Loading helper that makes loading config / dataset / model easier. This
    includes:
        `config_is_wavegan`, `load_dataset`, `load_dataset_wavegan`,
        `load_config`, `load_model`, `restore_model`.

  - Model helpers that makes model-related actions such as running,
    classifying and inferencing easier. This includes:
        `run_with_batch`, `ModelHelper`, `ModelWaveGANHelper`, `OneSideHelper`.

  - Miscellaneous Helpers, including
        `prepare_dirs`

"""
import importlib
import os

from magenta.models.latent_transfer import common
from magenta.models.latent_transfer import model_dataspace
import numpy as np
from scipy.io import wavfile
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string(
    'wavegan_gen_ckpt_dir', '', 'The directory to WaveGAN generator\'s ckpt. '
    'If WaveGAN is involved, this argument must be set.')
tf.flags.DEFINE_string(
    'wavegan_inception_ckpt_dir', '',
    'The directory to WaveGAN inception (classifier)\'s ckpt. '
    'If WaveGAN is involved, this argument must be set.')
tf.flags.DEFINE_string(
    'wavegan_latent_dir', '', 'The directory to WaveGAN\'s latent space.'
    'If WaveGAN is involved, this argument must be set.')


class BatchIndexIterator(object):
  """An inifite iterator each time yielding batch.

  This iterator yields the index of data instances rather than data itself.
  This design enables the index to be resuable in indexing multiple arrays.

  Args:
    n: An integer indicating total size of dataset.
    batch_size: An integer indicating size of batch.
  """

  def __init__(self, n, batch_size):
    """Inits this integer."""
    self.n = n
    self.batch_size = batch_size

    self._pos = 0
    self._order = self._make_random_order()

  def __iter__(self):
    return self

  def next(self):
    return self.__next__()

  def __next__(self):
    batch = []
    for i in range(self._pos, self._pos + self.batch_size):
      if i % self.n == 0:
        self._order = self._make_random_order()
      batch.append(self._order[i % self.n])
    batch = np.array(batch, dtype=np.int32)

    self._pos += self.batch_size
    return batch

  def _make_random_order(self):
    """Make a new, shuffled order."""
    return np.random.permutation(np.arange(0, self.n))


class InterGroupSamplingIndexIterator(object):
  """Radonmly samples index with a label group.

  This iterator yields a pair of indices in two dataset that always has the
  same label. This design enables the index to be resuable in indexing multiple
  arrays and is needed for the scenario where only label-level alignment is
  provided.

  Args:
    group_by_label_A: List of lists for data space A. The i-th list indicates
        the non-empty list of indices for data instance with i-th (zero-based)
        label.
    group_by_label_B: List of lists for data space B. The i-th list indicates
        the non-empty list of indices for data instance with i-th (zero-based)
        label.
    pairing_number: An integer indicating the umber of paired data to be used.
    batch_size: An integer indicating size of batch.
  """

  # Variable that in its name has A or B indicating their belonging of one side
  # of data has name consider to be invalid by pylint so we disable the warning.
  # pylint:disable=invalid-name
  def __init__(self, group_by_label_A, group_by_label_B, pairing_number,
               batch_size):
    assert len(group_by_label_A) == len(group_by_label_B)
    for _ in group_by_label_A:
      assert _
    for _ in group_by_label_B:
      assert _

    n_label = self.n_label = len(group_by_label_A)

    for i in range(n_label):
      if pairing_number >= 0:
        n_use = pairing_number // n_label
        if pairing_number % n_label != 0:
          n_use += int(i < pairing_number % n_label)
      else:
        n_use = max(len(group_by_label_A[i]), len(group_by_label_B[i]))
      group_by_label_A[i] = np.array(group_by_label_A[i])[:n_use]
      group_by_label_B[i] = np.array(group_by_label_B[i])[:n_use]
    self.group_by_label_A = group_by_label_A
    self.group_by_label_B = group_by_label_B
    self.batch_size = batch_size

    self._pos = 0

    self._sub_pos_A = [0] * n_label
    self._sub_pos_B = [0] * n_label

  def __iter__(self):
    return self

  def next(self):
    """Python 2 compatible interface."""
    return self.__next__()

  def __next__(self):
    batch = []
    for i in range(self._pos, self._pos + self.batch_size):
      label = i % self.n_label
      index_A = self.pick_index(self._sub_pos_A, self.group_by_label_A, label)
      index_B = self.pick_index(self._sub_pos_B, self.group_by_label_B, label)
      batch.append((index_A, index_B))
    batch = np.array(batch, dtype=np.int32)

    self._pos += self.batch_size
    return batch

  def pick_index(self, sub_pos, group_by_label, label):
    if sub_pos[label] == 0:
      np.random.shuffle(group_by_label[label])

    result = group_by_label[label][sub_pos[label]]
    sub_pos[label] = (sub_pos[label] + 1) % len(group_by_label[label])
    return result

  # pylint:enable=invalid-name


class GuasssianDataHelper(object):
  """A helper to hold data where each instance is a sampled point.

  Args:
    mu: Mean of data points.
    sigma: Variance of data points. If it is None, it is treated as zeros.
    batch_size: An integer indicating size of batch.
  """

  def __init__(self, mu, sigma=None):
    if sigma is None:
      sigma = np.zeros_like(mu)
    assert mu.shape == sigma.shape
    self.mu, self.sigma = mu, sigma

  def pick_batch(self, batch_index):
    """Pick a batch where instances are sampled from Guassian distributions."""
    mu, sigma = self.mu, self.sigma
    batch_mu, batch_sigma = self._np_index_arrs(batch_index, mu, sigma)
    batch = self._np_sample_from_gaussian(batch_mu, batch_sigma)
    return batch

  def __len__(self):
    return len(self.mu)

  @staticmethod
  def _np_sample_from_gaussian(mu, sigma):
    """Sampling from Guassian distribtuion specified by `mu` and `sigma`."""
    assert mu.shape == sigma.shape
    return mu + sigma * np.random.randn(*sigma.shape)

  @staticmethod
  def _np_index_arrs(index, *args):
    """Index arrays with the same given `index`."""
    return (arr[index] for arr in args)


class SingleDataIterator(object):
  """Iterator of a single-side dataset of encoded representation.

  Args:
    mu: Mean of data points.
    sigma: Variance of data points. If it is None, it is treated as zeros.
    batch_size: An integer indicating size of batch.
  """

  def __init__(self, mu, sigma, batch_size):
    self.data_helper = GuasssianDataHelper(mu, sigma)

    n = len(self.data_helper)
    self.batch_index_iterator = BatchIndexIterator(n, batch_size)

  def __iter__(self):
    return self

  def next(self):
    """Python 2 compatible interface."""
    return self.__next__()

  def __next__(self):
    batch_index = next(self.batch_index_iterator)
    batch = self.data_helper.pick_batch(batch_index)
    debug_info = (batch_index,)
    return batch, debug_info


class PairedDataIterator(object):
  """Iterator of a paired dataset of encoded representation.


  Args:
    mu_A: Mean of data points in data space A.
    sigma_A: Variance of data points in data space A. If it is None, it is
        treated as zeros.
    label_A: A List of labels for data points in data space A.
    index_grouped_by_label_A: List of lists for data space A. The i-th list
        indicates the non-empty list of indices for data instance with i-th
        (zero-based) label.
    mu_B: Mean of data points in data space B.
    sigma_B: Variance of data points in data space B. If it is None, it is
        treated as zeros.
    label_B: A List of labels for data points in data space B.
    index_grouped_by_label_B: List of lists for data space B. The i-th list
        indicates the non-empty list of indices for data instance with i-th
        (zero-based) label.
    pairing_number: An integer indicating the umber of paired data to be used.
    batch_size: An integer indicating size of batch.
  """

  # Variable that in its name has A or B indicating their belonging of one side
  # of data has name consider to be invalid by pylint so we disable the warning.
  # pylint:disable=invalid-name

  def __init__(self, mu_A, sigma_A, train_data_A, label_A,
               index_grouped_by_label_A, mu_B, sigma_B, train_data_B, label_B,
               index_grouped_by_label_B, pairing_number, batch_size):
    self._data_helper_A = GuasssianDataHelper(mu_A, sigma_A)
    self._data_helper_B = GuasssianDataHelper(mu_B, sigma_B)

    self.batch_index_iterator = InterGroupSamplingIndexIterator(
        index_grouped_by_label_A,
        index_grouped_by_label_B,
        pairing_number,
        batch_size,
    )

    self.label_A, self.label_B = label_A, label_B
    self.train_data_A, self.train_data_B = train_data_A, train_data_B

  def __iter__(self):
    return self

  def next(self):
    """Python 2 compatible interface."""
    return self.__next__()

  def __next__(self):
    batch_index = next(self.batch_index_iterator)
    batch_index_A, batch_index_B = (batch_index[:, 0], batch_index[:, 1])

    batch_A = self._data_helper_A.pick_batch(batch_index_A)
    batch_B = self._data_helper_B.pick_batch(batch_index_B)

    batch_label_A = self.label_A[batch_index_A]
    batch_label_B = self.label_B[batch_index_B]
    assert np.array_equal(batch_label_A, batch_label_B)

    batch_train_data_A = (
        None if self._train_data_A is None else self.train_data_A[batch_index_A]
    )
    batch_train_data_B = (
        None if self._train_data_B is None else self.train_data_B[batch_index_B]
    )
    debug_info = (batch_train_data_A, batch_train_data_B)

    return batch_A, batch_B, debug_info

  # pylint:enable=invalid-name


class ManualSummaryHelper(object):
  """A helper making manual TF summary easier."""

  def __init__(self):
    self._key_to_ph_summary_tuple = {}

  def get_summary(self, sess, key, value):
    """Get TF (scalar) summary.

    Args:
      sess: A TF Session to be used in making summary.
      key: A string indicating the name of summary.
      value: A string indicating the value of summary.

    Returns:
      A TF summary.
    """
    self._add_key_if_not_exists(key)
    placeholder, summary = self._key_to_ph_summary_tuple[key]
    return sess.run(summary, {placeholder: value})

  def _add_key_if_not_exists(self, key):
    """Add related TF heads for a key if it is not used before."""
    if key in self._key_to_ph_summary_tuple:
      return
    placeholder = tf.placeholder(tf.float32, shape=(), name=key + '_ph')
    summary = tf.summary.scalar(key, placeholder)
    self._key_to_ph_summary_tuple[key] = (placeholder, summary)


def config_is_wavegan(config):
  return config['dataset'].lower() == 'wavegan'


def load_dataset(config_name, exp_uid):
  """Load a dataset from a config's name.

  The loaded dataset consists of:
    - original data (dataset_blob, train_data, train_label),
    - encoded data from a pretrained model (train_mu, train_sigma), and
    - index grouped by label (index_grouped_by_label).

  Args:
    config_name: A string indicating the name of config to parameterize the
        model that associates with the dataset.
    exp_uid: A string representing the unique id of experiment to be used in
        model that associates with the dataset.

  Returns:
    An tuple of abovementioned components in the dataset.
  """

  config = load_config(config_name)
  if config_is_wavegan(config):
    return load_dataset_wavegan()

  model_uid = common.get_model_uid(config_name, exp_uid)

  dataset = common.load_dataset(config)
  train_data = dataset.train_data
  attr_train = dataset.attr_train
  path_train = os.path.join(dataset.basepath, 'encoded', model_uid,
                            'encoded_train_data.npz')
  train = np.load(path_train)
  train_mu = train['mu']
  train_sigma = train['sigma']
  train_label = np.argmax(attr_train, axis=-1)  # from one-hot to label
  index_grouped_by_label = common.get_index_grouped_by_label(train_label)

  tf.logging.info('index_grouped_by_label size: %s',
                  [len(_) for _ in index_grouped_by_label])

  tf.logging.info('train loaded from %s', path_train)
  tf.logging.info('train shapes: mu = %s, sigma = %s', train_mu.shape,
                  train_sigma.shape)
  dataset_blob = dataset
  return (dataset_blob, train_data, train_label, train_mu, train_sigma,
          index_grouped_by_label)


def load_dataset_wavegan():
  """Load WaveGAN's dataset.

  The loaded dataset consists of:
    - original data (dataset_blob, train_data, train_label),
    - encoded data from a pretrained model (train_mu, train_sigma), and
    - index grouped by label (index_grouped_by_label).

  Some of these attributes are not avaiable (set as None) but are left here
  to keep everything aligned with returned value of `load_dataset`.

  Returns:
    An tuple of abovementioned components in the dataset.
  """

  latent_dir = os.path.expanduser(FLAGS.wavegan_latent_dir)
  path_train = os.path.join(latent_dir, 'data_train.npz')
  train = np.load(path_train)
  train_z = train['z']
  train_label = train['label']
  index_grouped_by_label = common.get_index_grouped_by_label(train_label)

  dataset_blob, train_data = None, None
  train_mu, train_sigma = train_z, None
  return (dataset_blob, train_data, train_label, train_mu, train_sigma,
          index_grouped_by_label)


def load_config(config_name):
  """Load the config from its name."""
  return importlib.import_module('configs.%s' % config_name).config


def load_model(model_cls, config_name, exp_uid):
  """Load a model.

  Args:
    model_cls: A sonnet Class that is the factory of model.
    config_name: A string indicating the name of config to parameterize the
        model.
    exp_uid: A string representing the unique id of experiment to be used in
        model.

  Returns:
    An instance of sonnet model.
  """

  config = load_config(config_name)
  model_uid = common.get_model_uid(config_name, exp_uid)

  m = model_cls(config, name=model_uid)
  m()
  return m


def restore_model(saver, config_name, exp_uid, sess, save_path,
                  ckpt_filename_template):
  model_uid = common.get_model_uid(config_name, exp_uid)
  saver.restore(
      sess,
      os.path.join(
          save_path, model_uid, 'best', ckpt_filename_template % model_uid))


def prepare_dirs(
    signature='unspecified_signature',
    config_name='unspecified_config_name',
    exp_uid='unspecified_exp_uid',
):
  """Prepare saving and sampling direcotories for training.

  Args:
    signature: A string of signature of model such as `joint_model`.
    config_name: A string representing the name of config for joint model.
    exp_uid: A string representing the unique id of experiment to be used in
        joint model.

  Returns:
    A tuple of (save_dir, sample_dir). They are strings and are paths to the
        directory for saving checkpoints / summaries and path to the directory
        for saving samplings, respectively.
  """

  model_uid = common.get_model_uid(config_name, exp_uid)

  local_base_path = os.path.join(common.get_default_scratch(), signature)

  save_dir = os.path.join(local_base_path, 'ckpts', model_uid)
  tf.gfile.MakeDirs(save_dir)
  sample_dir = os.path.join(local_base_path, 'sample', model_uid)
  tf.gfile.MakeDirs(sample_dir)

  return save_dir, sample_dir


def run_with_batch(sess, op_target, op_feed, arr_feed, batch_size=None):
  if batch_size is None:
    batch_size = len(arr_feed)
  return np.concatenate([
      sess.run(op_target, {op_feed: arr_feed[i:i + batch_size]})
      for i in range(0, len(arr_feed), batch_size)
  ])


class ModelHelper(object):
  """A Helper that provides sampling and classification for pre-trained WaveGAN.

  This generic helper is for VAE model we trained as dataspace model.
  For external sourced model use specified helper such as `ModelWaveGANHelper`.
  """
  DEFAULT_BATCH_SIZE = 100

  def __init__(self, config_name, exp_uid):
    self.config_name = config_name
    self.exp_uid = exp_uid

    self.build()

  def build(self):
    """Build the TF graph and heads for dataspace model.

    It also prepares different graph, session and heads for sampling and
    classification respectively.
    """

    config_name = self.config_name
    config = load_config(config_name)
    exp_uid = self.exp_uid

    graph = tf.Graph()
    with graph.as_default():
      sess = tf.Session(graph=graph)
      m = load_model(model_dataspace.Model, config_name, exp_uid)

    self.config = config
    self.graph = graph
    self.sess = sess
    self.m = m

  def restore_best(self, saver_name, save_path, ckpt_filename_template):
    """Restore the weights of best pre-trained models."""
    config_name = self.config_name
    exp_uid = self.exp_uid
    sess = self.sess
    saver = getattr(self.m, saver_name)
    restore_model(saver, config_name, exp_uid, sess, save_path,
                  ckpt_filename_template)

  def decode(self, z, batch_size=None):
    """Decode from given latant space vectors `z`.

    Args:
      z: A numpy array of latent space vectors.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the sampling requires lots of GPU memory.

    Returns:
      A numpy array, the dataspace points from decoding.
    """
    m = self.m
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    return run_with_batch(self.sess, m.x_mean, m.z, z, batch_size)

  def classify(self, real_x, batch_size=None):
    """Classify given dataspace points `real_x`.

    Args:
      real_x: A numpy array of dataspace points.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the classification requires lots of GPU memory.

    Returns:
      A numpy array, the prediction from classifier.
    """
    m = self.m
    op_target = m.pred_classifier
    op_feed = m.x
    arr_feed = real_x
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    pred = run_with_batch(self.sess, op_target, op_feed, arr_feed, batch_size)
    pred = np.argmax(pred, axis=-1)
    return pred

  def save_data(self, x, name, save_dir, x_is_real_x=False):
    """Save dataspace instances.

    Args:
      x: A numpy array of dataspace points.
      name: A string indicating the name in the saved file.
      save_dir: A string indicating the directory to put the saved file.
      x_is_real_x: An boolean indicating whether `x` is already in dataspace. If
          not, `x` is converted to dataspace before saving
    """
    real_x = x if x_is_real_x else self.decode(x)
    real_x = common.post_proc(real_x, self.config)
    batched_real_x = common.batch_image(real_x)
    sample_file = os.path.join(save_dir, '%s.png' % name)
    common.save_image(batched_real_x, sample_file)


class ModelWaveGANHelper(object):
  """A Helper that provides sampling and classification for pre-trained WaveGAN.
  """
  DEFAULT_BATCH_SIZE = 100

  def __init__(self):
    self.build()

  def build(self):
    """Build the TF graph and heads from pre-trained WaveGAN ckpts.

    It also prepares different graph, session and heads for sampling and
    classification respectively.
    """

    # pylint:disable=unused-variable,possibly-unused-variable
    # Reason:
    #   All endpoints are stored as attribute at the end of `_build`.
    #   Pylint cannot infer this case so it emits false alarm of
    #   unused-variable if we do not disable this warning.

    # pylint:disable=invalid-name
    # Reason:
    #   Variable useing 'G' in is name to be consistent with WaveGAN's author
    #   has name consider to be invalid by pylint so we disable the warning.

    # Dataset (SC09, WaveGAN)'s generator
    graph_sc09_gan = tf.Graph()
    with graph_sc09_gan.as_default():
      # Use the retrained, Gaussian priored model
      gen_ckpt_dir = os.path.expanduser(FLAGS.wavegan_gen_ckpt_dir)
      sess_sc09_gan = tf.Session(graph=graph_sc09_gan)
      saver_gan = tf.train.import_meta_graph(
          os.path.join(gen_ckpt_dir, 'infer', 'infer.meta'))

    # Dataset (SC09, WaveGAN)'s  classifier (inception)
    graph_sc09_class = tf.Graph()
    with graph_sc09_class.as_default():
      inception_ckpt_dir = os.path.expanduser(FLAGS.wavegan_inception_ckpt_dir)
      sess_sc09_class = tf.Session(graph=graph_sc09_class)
      saver_class = tf.train.import_meta_graph(
          os.path.join(inception_ckpt_dir, 'infer.meta'))

    # Dataset B (SC09, WaveGAN)'s Tensor symbols
    sc09_gan_z = graph_sc09_gan.get_tensor_by_name('z:0')
    sc09_gan_G_z = graph_sc09_gan.get_tensor_by_name('G_z:0')[:, :, 0]

    # Classification: Tensor symbols
    sc09_class_x = graph_sc09_class.get_tensor_by_name('x:0')
    sc09_class_scores = graph_sc09_class.get_tensor_by_name('scores:0')

    # Add all endpoints as object attributes
    for k, v in locals().items():
      self.__dict__[k] = v

  def restore(self):
    """Restore the weights of models."""
    gen_ckpt_dir = self.gen_ckpt_dir
    graph_sc09_gan = self.graph_sc09_gan
    saver_gan = self.saver_gan
    sess_sc09_gan = self.sess_sc09_gan

    inception_ckpt_dir = self.inception_ckpt_dir
    graph_sc09_class = self.graph_sc09_class
    saver_class = self.saver_class
    sess_sc09_class = self.sess_sc09_class

    with graph_sc09_gan.as_default():
      saver_gan.restore(
          sess_sc09_gan,
          os.path.join(gen_ckpt_dir, 'bridge', 'model.ckpt'))

    with graph_sc09_class.as_default():
      saver_class.restore(sess_sc09_class,
                          os.path.join(inception_ckpt_dir, 'best_acc-103005'))

    # pylint:enable=unused-variable,possibly-unused-variable
    # pylint:enable=invalid-name

  def decode(self, z, batch_size=None):
    """Decode from given latant space vectors `z`.

    Args:
      z: A numpy array of latent space vectors.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the sampling requires lots of GPU memory.

    Returns:
      A numpy array, the dataspace points from decoding.
    """
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    return run_with_batch(self.sess_sc09_gan, self.sc09_gan_G_z,
                          self.sc09_gan_z, z, batch_size)

  def classify(self, real_x, batch_size=None):
    """Classify given dataspace points `real_x`.

    Args:
      real_x: A numpy array of dataspace points.
      batch_size: (Optional) a integer to indication batch size for computation
          which is useful if the classification requires lots of GPU memory.

    Returns:
      A numpy array, the prediction from classifier.
    """
    batch_size = batch_size or self.DEFAULT_BATCH_SIZE
    pred = run_with_batch(self.sess_sc09_class, self.sc09_class_scores,
                          self.sc09_class_x, real_x, batch_size)
    pred = np.argmax(pred, axis=-1)
    return pred

  def save_data(self, x, name, save_dir, x_is_real_x=False):
    """Save dataspace instances.

    Args:
      x: A numpy array of dataspace points.
      name: A string indicating the name in the saved file.
      save_dir: A string indicating the directory to put the saved file.
      x_is_real_x: An boolean indicating whether `x` is already in dataspace. If
          not, `x` is converted to dataspace before saving
    """
    real_x = x if x_is_real_x else self.decode(x)
    real_x = real_x.reshape(-1)
    sample_file = os.path.join(save_dir, '%s.wav' % name)
    wavfile.write(sample_file, rate=16000, data=real_x)


class OneSideHelper(object):
  """The helper that manages model and classifier in dataspace for joint model.

  Args:
    config_name: A string representing the name of config for model in
        dataspace.
    exp_uid: A string representing the unique id of experiment used in
        the model in dataspace.
    config_name_classifier: A string representing the name of config for
        clasisifer in dataspace.
    exp_uid_classifier: A string representing the unique id of experiment used
        in the clasisifer in dataspace.
  """

  def __init__(
      self,
      config_name,
      exp_uid,
      config_name_classifier,
      exp_uid_classifier,
  ):
    config = load_config(config_name)
    this_config_is_wavegan = config_is_wavegan(config)
    if this_config_is_wavegan:
      # The sample object servers both purpose.
      m_helper = ModelWaveGANHelper()
      m_classifier_helper = m_helper
    else:
      # In this case two diffent objects serve two purpose.
      m_helper = ModelHelper(config_name, exp_uid)
      m_classifier_helper = ModelHelper(config_name_classifier,
                                        exp_uid_classifier)

    self.config_name = config_name
    self.this_config_is_wavegan = this_config_is_wavegan
    self.config = config
    self.m_helper = m_helper
    self.m_classifier_helper = m_classifier_helper

  def restore(self, dataset_blob):
    """Restore the pretrained model and classifier.

    Args:
      dataset_blob: The object containts `save_path` used for restoring.
    """
    this_config_is_wavegan = self.this_config_is_wavegan
    m_helper = self.m_helper
    m_classifier_helper = self.m_classifier_helper

    if this_config_is_wavegan:
      m_helper.restore()
      # We don't need restore the `m_classifier_helper` again since `m_helper`
      # and `m_classifier_helper` are two identicial objects.
    else:
      m_helper.restore_best('vae_saver', dataset_blob.save_path,
                            'vae_best_%s.ckpt')
      m_classifier_helper.restore_best(
          'classifier_saver', dataset_blob.save_path, 'classifier_best_%s.ckpt')
