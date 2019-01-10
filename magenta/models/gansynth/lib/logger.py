# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logger class to assist writing logs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import StringIO

import numpy as np
import scipy.io.wavfile
import scipy.misc
import tensorflow as tf


class Logger(object):
  """Wrapper for tf.summary.FileWriter that makes some things easier.

  Features:
  - explicit control of when logs are written
  - supports adding data without tensor ops, e.g. logger.add_scalar('cost', 5.0)
  - convenience method for saving sample sheets (grids of images)
  - can print info to stdout as well
  """

  def __init__(self, output_dir):
    super(Logger, self).__init__()
    # Create the directory if it doesn't exist already
    if output_dir is not None:
      self._output_dir = output_dir
      tf.gfile.MakeDirs(output_dir)
      self._writer = tf.summary.FileWriter(output_dir,
                                           max_queue=9999999,
                                           flush_secs=9999999)
    else:
      print('Warning: Logger instantiated without an output dir. '
            'Only printing to console.')
      self._writer = None
    self._since_last_print = {}
    self._summary_buffer = []

  def print(self, step):
    prints = ['{}\t{}'.format(name, val)
              for name, val in self._since_last_print.items()]
    to_print = ('iter {}\t{}'.format(step, '\t'.join(prints)))
    print(to_print)
    print('WARNING logging to: '+self._output_dir+'/logfile')
    with tf.gfile.GFile(self._output_dir+'/logfile', 'a') as f:
      f.write(to_print+'\n')
    self._since_last_print = {}

  def flush(self):
    if self._writer:
      for summary, step in self._summary_buffer:
        self._writer.add_summary(summary, step)
      self._writer.flush()
    self._summary_buffer = []

  def add_summary(self, summary, step):
    self._summary_buffer.append((summary, step))

  def add_scalar(self, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self._since_last_print[tag] = value
    self.add_summary(summary, step)

  def add_image(self, tag, image, step):
    """image: HWC uint8 numpy array."""
    s = StringIO.StringIO()
    scipy.misc.imsave(s, image, format='png')
    summary_image = tf.Summary.Image(
        encoded_image_string=s.getvalue(),
        height=image.shape[0], width=image.shape[1])
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=summary_image)])
    self._since_last_print[tag] = '(image)'
    self.add_summary(summary, step)

  def add_image_grid(self, tag, images, step):
    """images: BHWC uint8 numpy array."""
    # Calculate number of rows / cols
    n_samples = images.shape[0]
    n_rows = int(np.sqrt(n_samples))
    while n_samples % n_rows != 0:
      n_rows -= 1
    n_cols = n_samples//n_rows

    # Copy each image into its spot in the grid
    height, width = images[0].shape[:2]
    grid_image = np.zeros((height*n_rows, width*n_cols, 3), dtype='uint8')
    for n, image in enumerate(images):
      j = n // n_cols
      i = n % n_cols
      grid_image[j*height:j*height+height, i*width:i*width+width] = image

    self.add_image(tag, grid_image, step)

  def add_audio(self, tag, audio, step, sample_rate=16000):
    """audio: array of float32 at 16KHz."""
    # normalize to max(abs(audio))=1
    audio /= np.max(np.abs(audio))
    s = StringIO.StringIO()
    scipy.io.wavfile.write(s, sample_rate, audio)
    summary_audio = tf.Summary.Audio(
        sample_rate=sample_rate,
        num_channels=1,
        length_frames=len(audio),
        encoded_audio_string=s.getvalue(),
        content_type='audio/wav'
    )
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, audio=summary_audio)])
    self._since_last_print[tag] = '(audio)'
    self.add_summary(summary, step)
