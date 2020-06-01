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
"""Sample from pre-trained WaveGAN model.

This script provides sampling from pre-trained WaveGAN model that is done
through the original author's code (https://github.com/chrisdonahue/wavegan).
The main purpose is to help manually check the quality of WaveGAN model.
"""

import operator
import os

import numpy as np
from scipy.io import wavfile
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('total_per_label', '7000',
                        'Minimal # samples per label')
tf.flags.DEFINE_integer('top_per_label', '1700', '# of top samples per label')
tf.flags.DEFINE_string('gen_ckpt_dir', '',
                       'The directory to WaveGAN generator\'s ckpt.')
tf.flags.DEFINE_string(
    'inception_ckpt_dir', '',
    'The directory to WaveGAN inception (classifier)\'s ckpt.')
tf.flags.DEFINE_string('latent_dir', '',
                       'The directory to WaveGAN\'s latent space.')


def main(unused_argv):

  # pylint:disable=invalid-name
  # Reason:
  #   Following variables have their name consider to be invalid by pylint so
  #   we disable the warning.
  #   - Variable that is class

  del unused_argv

  use_gaussian_pretrained_model = FLAGS.use_gaussian_pretrained_model

  gen_ckpt_dir = FLAGS.gen_ckpt_dir
  inception_ckpt_dir = FLAGS.inception_ckpt_dir

  # TF init
  tf.reset_default_graph()
  # - generative model
  graph_gan = tf.Graph()
  with graph_gan.as_default():
    sess_gan = tf.Session(graph=graph_gan)
    if use_gaussian_pretrained_model:
      saver_gan = tf.train.import_meta_graph(
          os.path.join(gen_ckpt_dir, '..', 'infer', 'infer.meta'))
      saver_gan.restore(sess_gan, os.path.join(gen_ckpt_dir, 'model.ckpt'))
    else:
      saver_gan = tf.train.import_meta_graph(
          os.path.join(gen_ckpt_dir, 'infer.meta'))
      saver_gan.restore(sess_gan, os.path.join(gen_ckpt_dir, 'model.ckpt'))
  # - classifier (inception)
  graph_class = tf.Graph()
  with graph_class.as_default():
    sess_class = tf.Session(graph=graph_class)
    saver_class = tf.train.import_meta_graph(
        os.path.join(inception_ckpt_dir, 'infer.meta'))
    saver_class.restore(
        sess_class, os.path.join(inception_ckpt_dir, 'best_acc-103005'))

  # Generate: Tensor symbols
  z = graph_gan.get_tensor_by_name('z:0')
  G_z = graph_gan.get_tensor_by_name('G_z:0')[:, :, 0]
  # G_z_spec = graph_gan.get_tensor_by_name('G_z_spec:0')
  # Classification: Tensor symbols
  x = graph_class.get_tensor_by_name('x:0')
  scores = graph_class.get_tensor_by_name('scores:0')

  # Sample something AND classify them

  output_dir = FLAGS.latent_dir

  tf.gfile.MakeDirs(output_dir)

  np.random.seed(19260817)
  total_per_label = FLAGS.total_per_label
  top_per_label = FLAGS.top_per_label
  group_by_label = [[] for _ in range(10)]
  batch_size = 200
  hidden_dim = 100

  with tqdm(desc='min label count', unit=' #', total=total_per_label) as pbar:
    label_count = [0] * 10
    last_min_label_count = 0
    while True:
      min_label_count = min(label_count)
      pbar.update(min_label_count - last_min_label_count)
      last_min_label_count = min_label_count

      if use_gaussian_pretrained_model:
        _z = np.random.randn(batch_size, hidden_dim)
      else:
        _z = (np.random.rand(batch_size, hidden_dim) * 2.) - 1.
      # _G_z, _G_z_spec = sess_gan.run([G_z, G_z_spec], {z: _z})
      _G_z = sess_gan.run(G_z, {z: _z})
      _x = _G_z
      _scores = sess_class.run(scores, {x: _x})
      _max_scores = np.max(_scores, axis=1)
      _labels = np.argmax(_scores, axis=1)
      for i in range(batch_size):
        label = _labels[i]

        group_by_label[label].append((_max_scores[i], (_z[i], _G_z[i])))
        label_count[label] += 1

        if len(group_by_label[label]) >= top_per_label * 2:
          # remove unneeded tails
          group_by_label[label].sort(key=operator.itemgetter(0), reverse=True)
          group_by_label[label] = group_by_label[label][:top_per_label]

      if last_min_label_count >= total_per_label:
        break

  for label in range(10):
    group_by_label[label].sort(key=operator.itemgetter(0), reverse=True)
    group_by_label[label] = group_by_label[label][:top_per_label]

  # output a few samples as image
  image_output_dir = os.path.join(output_dir, 'sample_iamge')
  tf.gfile.MakeDirs(image_output_dir)

  for label in range(10):
    group_by_label[label].sort(key=operator.itemgetter(0), reverse=True)
    index = 0
    for confidence, (
        _,
        this_G_z,
    ) in group_by_label[label][:10]:
      output_basename = 'predlabel=%d_index=%02d_confidence=%.6f' % (
          label, index, confidence)
      wavfile.write(
          filename=os.path.join(
              image_output_dir, output_basename + '_sound.wav'),
          rate=16000,
          data=this_G_z)

  # Make Numpy arrays and save everything as an npz file
  array_label, array_z, array_G_z = [], [], []
  for label in range(10):
    for _, blob in group_by_label[label]:
      this_z, this_G_z = blob[:2]
      array_label.append(label)
      array_z.append(this_z)
      array_G_z.append(this_G_z)
  array_label = np.array(array_label, dtype='i')
  array_z = np.array(array_z)
  array_G_z = np.array(array_G_z)

  np.savez(
      os.path.join(output_dir, 'data_train.npz'),
      label=array_label,
      z=array_z,
      G_z=array_G_z,
  )

  # pylint:enable=invalid-name


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.app.run(main)
