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
r"""Run all evaluations on a given model.

1. Evaluation of model-checkpoint:

blaze run evaluate -- \
  "{\"weights_path\":\"<path-to-weight-dir>\", \"dataset_name\":\"full\"}"

########################################################################

2. Generate samples for a model:

blaze run evaluate -- \
  "{\"weights_path\":\"<path-to-weight-dir>\", \"gen_samples\":\"True\",
  \"fake_samples_path\":\"<savepath-for-samples>\"}"

########################################################################

3. Evaluate model from samples:

blaze run evaluate -- \
  "{\"fake_samples_path\":\"<path-to-samples-dir>\", \"use_samples\":\"True\",
  \"dataset_name\":\"acoustic_only\", \"output_dir\":\"<path-to-logs>\"}"

NOTE: Currently, the samples should be separate in .wav files structured as
      <path-to-samples-dir>/pitch_<pitch-id>/sample_<sample-id>.wav
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.flags
import tensorflow as tf

from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import logger as lib_logger
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib.datasets import dataset_nsynth_tfrecord
from magenta.models.gansynth.lib.eval import eval_util
from magenta.models.gansynth.lib.eval import evaluate

absl.flags.DEFINE_string('hparams', None, 'Flags dict as JSON string')
FLAGS = absl.flags.FLAGS


def set_flags(flags):
  """Setting flags."""
  flags.set_if_empty('use_samples', True)
  flags.set_if_empty('gen_samples', False)
  flags.set_if_empty('with_replacement', False)
  flags.set_if_empty('run_evaluate', True)
  flags.set_if_empty('num_samples_per_pitch', 2000)
  flags.set_if_empty(
      'fake_samples_path',
      '/tmp/gansynth_fake_samples/')
  flags.set_if_empty(
      'real_samples_path',
      '/tmp/path-to-real-samples/')
  flags.set_if_empty(
      'weights_path',
      '/tmp/path-to-weights/')
  flags.set_if_empty('dataset_name', 'acoustic_eval')
  evaluate.set_flags(flags)


def main(_):
  flags = lib_flags.Flags()
  flags.load_json(FLAGS.hparams)
  set_flags(flags)

  print('Flags:')
  flags.print_values()

  fake_generate_samples = None

  if flags.gen_samples:
    print('Generating samples!')
    # Load model from train_root_dir.
    model_def = lib_model.Model
    model = model_def.load_from_path(flags.weights_path)
    fake_generate_samples = model.generate_samples

    # Generate and save samples.
    pitch_counts = dataset_nsynth_tfrecord.get_pitch_counts(
        dataset_name=flags.dataset_name)
    for pitch in pitch_counts.keys():
      samples = fake_generate_samples(flags.num_samples_per_pitch, pitch)
      eval_util.write_samples_to_file(pitch, samples, flags.fake_samples_path)

  # Run Evaluation.
  if flags.run_evaluate:
    print('Running evaluation!')
    # Read real dataset based on pitch_counters.
    real_generate_samples = eval_util.generate_samples_from_file(
        flags.real_samples_path,
        flags.dataset_name,
        flags.with_replacement)

    # If evaluating from files, override the generate_samples fn.
    if flags.use_samples:
      fake_generate_samples = eval_util.generate_samples_from_file(
          flags.fake_samples_path,
          flags.dataset_name,
          flags.with_replacement)
    elif not fake_generate_samples:
      model_def = lib_model.Model
      model = model_def.load_from_path(flags.weights_path)
      fake_generate_samples = model.generate_samples

    logger = lib_logger.Logger(flags.fake_samples_path)
    eval_results = evaluate.run_everything(
        flags, fake_generate_samples,
        real_generate_samples=real_generate_samples)
    for k, v in eval_results.items():
      logger.add_scalar(k, v, 0)
    logger.print(0)
    logger.flush()
    print('done!')

if __name__ == '__main__':
  tf.app.run()
