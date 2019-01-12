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
"""One run_everything() function that runs every evaluation together."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import datasets
from magenta.models.gansynth.lib.eval import classifier_score
from magenta.models.gansynth.lib.eval import eval_util
from magenta.models.gansynth.lib.eval import fid
from magenta.models.gansynth.lib.eval import mmd
from magenta.models.gansynth.lib.eval import nn_divergence
from magenta.models.gansynth.lib.eval import pitch_classifier


def set_flags(flags):
  """Set default flags for evaluation."""

  flags.set_if_empty('nn_divergence', lib_flags.Flags())
  flags.set_if_empty('skip_nn_div', False)
  nn_divergence.set_flags(flags.nn_divergence)

  ## For now num_samples should be multiple of batch_size
  flags.set_if_empty('eval_batch_size', 128)
  flags.set_if_empty('cs_num_samples', 12800)
  flags.set_if_empty('fid_num_samples', 12800)
  flags.set_if_empty('mmd_num_samples', 12800)
  flags.set_if_empty('pitch_accuracy_num_samples', 12800)
  flags.set_if_empty('dataset_name', 'acoustic_eval')  # 'full', 'acoustic_only'

  flags.set_if_empty('classifier_layer', 'preact_layer_7_1')
  flags.set_if_empty('max_pitch', 84)  # full is 120
  flags.set_if_empty('min_pitch', 24)  # full is 9
  flags.set_if_empty('mmd_kernel', 'polynomial')
  flags.set_if_empty('threshold', 1)

  ## Set default paths
  flags.set_if_empty(
      'pitch_checkpoint_dir',
      '/tmp/path-to-pretrained-pitch-model/')
  flags.set_if_empty(
      'real_samples_path',
      '/tmp/path-to-eval-samples/')
  flags.set_if_empty('with_replacement', False)


def _get_tensors(num_samples, generate_samples, flags,
                 pitch_counter=None,
                 return_pitch_counter=False):
  """Utility function for generating samples from model.

  Args:
    num_samples: Number of samples to generate
    generate_samples: generator function
    flags: flags for generative_model
    pitch_counter: if given, samples according to this distribution.
    return_pitch_counter: (boolean) if True returns pitch_counter
  Returns:
    data_samples: batch of `num_sample` data samples
    pitches: corresponding pitches for `fake_data`
    pitch_counter(optional): pitch_counter for this sampling step.
  """
  if pitch_counter is None:
    pitch_counter = datasets.get_pitches(num_samples,
                                         dataset_name=flags.dataset_name)
  idx = 0
  pitches = np.zeros(num_samples)
  data_samples = np.zeros((1, 64000), dtype='float32')
  for pitch, n_sample in pitch_counter.items():
    samples = generate_samples(n_sample, pitch)
    data_samples = np.concatenate((data_samples, samples), axis=0)
    pitches[idx: idx+n_sample] = pitch
    idx += n_sample
  if return_pitch_counter:
    return data_samples[1:, :], pitches, pitch_counter
  return data_samples[1:, :], pitches


def _run_nn_divergence(flags, generate_samples,
                       real_generate_samples=None):
  """Evaluates NN-Divergence.

  Args:
    flags: (dict) Evaluation specific flags.
    generate_samples: (function) generator_fn for fake samples.
    real_generate_samples: (function) generator_fn for real samples.

  Returns:
    results: {`nn_divergence`, `nn_divergence_std`}
  """

  if real_generate_samples is None:
    real_generate_samples = eval_util.generate_samples_from_file(
        flags.real_samples_path,
        flags.dataset_name,
        flags.with_replacement)

  print('Evaluating NN-Divergence!')
  real_samples = _get_tensors(10*1000, real_generate_samples, flags)[0]
  results = []
  for _ in xrange(5):
    fake_samples = _get_tensors(10*1000, generate_samples, flags)[0]
    results.append(nn_divergence.run(flags.nn_divergence,
                                     real_samples, fake_samples))
    return {'nn_divergence': np.mean(results),
            'nn_divergence_std': np.std(results)}


def _run_pitch_classifier(flags, generate_samples):
  """Evaluates pitch acuuracy / mean entropy distribution.

  Args:
    flags: (dict) Evaluation specific flags.
    generate_samples: (function) generator_fn for fake samples.

  Returns:
    results: {`mean_pitch_accuracy`, `mean_pitch_entropy`}
  """
  print('Evaluating pitch-accuracy!')
  num_samples = flags['pitch_accuracy_num_samples']
  samples, pitches = _get_tensors(num_samples, generate_samples, flags)
  pitches = np.squeeze(pitches)
  results = pitch_classifier.run(flags, samples, pitches)
  return results


def _run_mmd(flags, generate_samples, real_generate_samples=None):
  """Evaluate MMD.

  Args:
    flags: (dict) Evaluation specific flags.
    generate_samples: (function) generator_fn for fake samples.
    real_generate_samples: (function) generator_fn for real samples.

  Returns:
    results: MMD estimate.
  """

  if real_generate_samples is None:
    real_generate_samples = eval_util.generate_samples_from_file(
        flags.real_samples_path,
        flags.dataset_name,
        flags.with_replacement)

  print('Evaluating MMD!')
  num_samples = flags['mmd_num_samples']
  real_samples = _get_tensors(num_samples, real_generate_samples, flags)[0]
  fake_samples = _get_tensors(num_samples, generate_samples, flags)[0]
  results = mmd.run(flags, real_samples, fake_samples)
  return results


def _run_fid(flags, generate_samples, real_generate_samples=None):
  """Evaluate FID.

  Args:
    flags: (dict) Evaluation specific flags.
    generate_samples: (function) generator_fn for fake samples.
    real_generate_samples: (function) generator_fn for real samples.

  Returns:
    results: FID estimate.
  """

  if real_generate_samples is None:
    real_generate_samples = eval_util.generate_samples_from_file(
        flags.real_samples_path,
        flags.dataset_name,
        flags.with_replacement)

  print('Evaluating FID!')
  num_samples = flags['fid_num_samples']
  real_samples = _get_tensors(num_samples, real_generate_samples, flags)[0]
  fake_samples = _get_tensors(num_samples, generate_samples, flags)[0]
  results = fid.run(flags, real_samples, fake_samples)
  return results


def _run_classifier_score(flags, generate_samples):
  """Evaluate Classifier (inception-style) score.

  Args:
    flags: (dict) Evaluation specific flags.
    generate_samples: (function) generator_fn for fake samples.

  Returns:
    results: Classifier-Score estimate.
  """
  print('Evaluating Classifier-Score!')
  num_samples = flags['cs_num_samples']
  samples = _get_tensors(num_samples, generate_samples, flags)[0]
  results = classifier_score.run(flags, samples)
  return results


def run_everything(flags, generate_samples,
                   real_generate_samples=None):
  """Runs model evaluation across metrics.

  Args:
    flags: (dict) Evaluation specific flags.
    generate_samples: (function) generator_fn for fake samples.
    real_generate_samples: (function) generator_fn for real samples.
  Returns:
    results: (dict) Evaluation metrics.
  """
  if real_generate_samples is None:
    real_generate_samples = eval_util.generate_samples_from_file(
        flags.real_samples_path,
        flags.dataset_name,
        flags.with_replacement)

  print('Running model evaluation on %s dataset!' % flags.dataset_name)
  results = {}
  if not flags.skip_nn_div:
    results.update(_run_nn_divergence(flags, generate_samples,
                                      real_generate_samples=None))
  results.update(_run_pitch_classifier(flags, generate_samples))
  results.update(_run_mmd(flags, generate_samples,
                          real_generate_samples=None))
  results.update(_run_fid(flags, generate_samples,
                          real_generate_samples=None))
  results.update(_run_classifier_score(flags, generate_samples))
  return results
