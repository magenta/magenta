# Copyright 2016 Google Inc. All Rights Reserved.
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
""""Downloads classical MIDI files from midiworld.com.

Sample usage:
  $ bazel build magenta:download_midiworld_classical
  $ ./bazel-bin/magenta/download_midiworld_classical \
    --output_dir=~/magenta_data/midi/classical
"""

import logging
import os
import re
import sys
import tensorflow as tf
import urllib

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_dir', None,
                           'Directory where MIDI files will be saved to.')


def main(unused_argv):
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  root.addHandler(ch)

  if not FLAGS.output_dir:
    logging.fatal('--output_dir required')

  FLAGS.output_dir = os.path.expanduser(FLAGS.output_dir)

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  url = 'http://www.midiworld.com/classic.htm'
  midi_urls = re.findall(r'<a href="([^"]+.mid)">', urllib.urlopen(url).read())
  logging.info('Downloading midi files...')
  for midi_url in midi_urls:
    output_path = os.path.join(FLAGS.output_dir, os.path.basename(midi_url))
    urllib.urlretrieve(midi_url, output_path)
    logging.info(output_path)
  logging.info('Done.')


if __name__ == '__main__':
  tf.app.run()
