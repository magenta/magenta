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

"""Performance generation from score in Tensor2Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from magenta.models.score2perf import modalities
from magenta.models.score2perf import music_encoders
from note_seq import chord_symbols_lib
from note_seq import sequences_lib
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities as t2t_modalities
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

# Instead of importing datagen_beam (only needed for datagen) here, it is
# imported inline when needed to avoid the transitive apache_beam dependency
# when doing training or inference.

# TODO(iansimon): figure out the best way not to hard-code these constants
NUM_VELOCITY_BINS = 32
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108

# pylint: disable=line-too-long
MAESTRO_TFRECORD_PATHS = {
    'train': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_train.tfrecord',
    'dev': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_validation.tfrecord',
    'test': 'gs://magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0_test.tfrecord'
}
# pylint: enable=line-too-long


# Beam input transform for MAESTRO dataset.
def _maestro_input_transform():
  from magenta.models.score2perf import datagen_beam  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  return dict(
      (split_name, datagen_beam.ReadNoteSequencesFromTFRecord(tfrecord_path))
      for split_name, tfrecord_path in MAESTRO_TFRECORD_PATHS.items())


class Score2PerfProblem(problem.Problem):
  """Base class for musical score-to-performance problems.

  Data files contain tf.Example protos with encoded performance in 'targets' and
  optional encoded score in 'inputs'.
  """

  @property
  def splits(self):
    """Dictionary of split names and probabilities. Must sum to one."""
    raise NotImplementedError()

  @property
  def min_hop_size_seconds(self):
    """Minimum hop size in seconds at which to split input performances."""
    raise NotImplementedError()

  @property
  def max_hop_size_seconds(self):
    """Maximum hop size in seconds at which to split input performances."""
    raise NotImplementedError()

  @property
  def num_replications(self):
    """Number of times entire input performances will be split."""
    return 1

  @property
  def add_eos_symbol(self):
    """Whether to append EOS to encoded performances."""
    raise NotImplementedError()

  @property
  def absolute_timing(self):
    """Whether or not score should use absolute (vs. tempo-relative) timing."""
    return False

  @property
  def stretch_factors(self):
    """Temporal stretch factors for data augmentation (in datagen)."""
    return [1.0]

  @property
  def transpose_amounts(self):
    """Pitch transposition amounts for data augmentation (in datagen)."""
    return [0]

  @property
  def random_crop_length_in_datagen(self):
    """Randomly crop targets to this length in datagen."""
    return None

  @property
  def random_crop_in_train(self):
    """Whether to randomly crop each training example when preprocessing."""
    return False

  @property
  def split_in_eval(self):
    """Whether to split each eval example when preprocessing."""
    return False

  def performances_input_transform(self, tmp_dir):
    """Input performances beam transform (or dictionary thereof) for datagen."""
    raise NotImplementedError()

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    del task_id

    from magenta.models.score2perf import datagen_beam  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    def augment_note_sequence(ns, stretch_factor, transpose_amount):
      """Augment a NoteSequence by time stretch and pitch transposition."""
      augmented_ns = sequences_lib.stretch_note_sequence(
          ns, stretch_factor, in_place=False)
      try:
        _, num_deleted_notes = sequences_lib.transpose_note_sequence(
            augmented_ns, transpose_amount,
            min_allowed_pitch=MIN_PITCH, max_allowed_pitch=MAX_PITCH,
            in_place=True)
      except chord_symbols_lib.ChordSymbolError:
        raise datagen_beam.DataAugmentationError(
            'Transposition of chord symbol(s) failed.')
      if num_deleted_notes:
        raise datagen_beam.DataAugmentationError(
            'Transposition caused out-of-range pitch(es).')
      return augmented_ns

    augment_params = itertools.product(
        self.stretch_factors, self.transpose_amounts)
    augment_fns = [
        functools.partial(augment_note_sequence,
                          stretch_factor=s, transpose_amount=t)
        for s, t in augment_params
    ]

    datagen_beam.generate_examples(
        input_transform=self.performances_input_transform(tmp_dir),
        output_dir=data_dir,
        problem_name=self.dataset_filename(),
        splits=self.splits,
        min_hop_size_seconds=self.min_hop_size_seconds,
        max_hop_size_seconds=self.max_hop_size_seconds,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        num_replications=self.num_replications,
        encode_performance_fn=self.performance_encoder().encode_note_sequence,
        encode_score_fns=dict((name, encoder.encode_note_sequence)
                              for name, encoder in self.score_encoders()),
        augment_fns=augment_fns,
        absolute_timing=self.absolute_timing,
        random_crop_length=self.random_crop_length_in_datagen)

  def hparams(self, defaults, model_hparams):
    del model_hparams   # unused
    perf_encoder = self.get_feature_encoders()['targets']
    defaults.modality = {'targets': t2t_modalities.ModalityType.SYMBOL}
    defaults.vocab_size = {'targets': perf_encoder.vocab_size}
    if self.has_inputs:
      score_encoder = self.get_feature_encoders()['inputs']
      if isinstance(score_encoder.vocab_size, list):
        # TODO(trandustin): We default to not applying any transformation; to
        # apply one, pass modalities.bottom to the model's hparams.bottom. In
        # future, refactor the tuple of the "inputs" feature to be part of the
        # features dict itself, i.e., have multiple inputs each with its own
        # modality and vocab size.
        modality_cls = t2t_modalities.ModalityType.IDENTITY
      else:
        modality_cls = t2t_modalities.ModalityType.SYMBOL
      defaults.modality['inputs'] = modality_cls
      defaults.vocab_size['inputs'] = score_encoder.vocab_size

  def performance_encoder(self):
    """Encoder for target performances."""
    return music_encoders.MidiPerformanceEncoder(
        steps_per_second=STEPS_PER_SECOND,
        num_velocity_bins=NUM_VELOCITY_BINS,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        add_eos=self.add_eos_symbol)

  def score_encoders(self):
    """List of (name, encoder) tuples for input score components."""
    return []

  def feature_encoders(self, data_dir):
    del data_dir
    encoders = {
        'targets': self.performance_encoder()
    }
    score_encoders = self.score_encoders()
    if score_encoders:
      if len(score_encoders) > 1:
        # Create a composite score encoder, only used for inference.
        encoders['inputs'] = music_encoders.CompositeScoreEncoder(
            [encoder for _, encoder in score_encoders])
      else:
        # If only one score component, just use its encoder.
        _, encoders['inputs'] = score_encoders[0]
    return encoders

  def example_reading_spec(self):
    data_fields = {
        'targets': tf.VarLenFeature(tf.int64)
    }
    for name, _ in self.score_encoders():
      data_fields[name] = tf.VarLenFeature(tf.int64)

    # We don't actually "decode" anything here; the encodings are simply read as
    # tensors.
    data_items_to_decoders = None

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    if self.has_inputs:
      # Stack encoded score components depthwise as inputs.
      inputs = []
      for name, _ in self.score_encoders():
        inputs.append(tf.expand_dims(example[name], axis=1))
        del example[name]
      example['inputs'] = tf.stack(inputs, axis=2)

    if self.random_crop_in_train and mode == tf.estimator.ModeKeys.TRAIN:
      # Take a random crop of the training example.
      assert not self.has_inputs
      max_offset = tf.maximum(
          tf.shape(example['targets'])[0] - hparams.max_target_seq_length, 0)
      offset = tf.cond(
          max_offset > 0,
          lambda: tf.random_uniform([], maxval=max_offset, dtype=tf.int32),
          lambda: 0
      )
      example['targets'] = (
          example['targets'][offset:offset + hparams.max_target_seq_length])
      return example

    elif self.split_in_eval and mode == tf.estimator.ModeKeys.EVAL:
      # Split the example into non-overlapping segments.
      assert not self.has_inputs
      length = tf.shape(example['targets'])[0]
      extra_length = tf.mod(length, hparams.max_target_seq_length)
      examples = {
          'targets': tf.reshape(
              example['targets'][:length - extra_length],
              [-1, hparams.max_target_seq_length, 1, 1])
      }
      extra_example = {
          'targets': tf.reshape(
              example['targets'][-extra_length:], [1, -1, 1, 1])
      }
      dataset = tf.data.Dataset.from_tensor_slices(examples)
      extra_dataset = tf.data.Dataset.from_tensor_slices(extra_example)
      return dataset.concatenate(extra_dataset)

    else:
      # If not cropping or splitting, do standard preprocessing.
      return super(Score2PerfProblem, self).preprocess_example(
          example, mode, hparams)


class ConditionalScore2PerfProblem(Score2PerfProblem):
  """Lightweight version of base class for score-to-performance problems.

  This version incorporates one performance conditioning signal.
  Data files contain tf.Example protos with encoded performance in 'targets' and
  optional encoded score in 'inputs'.
  """

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    del task_id

    from magenta.models.score2perf import datagen_beam  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    def augment_note_sequence(ns, stretch_factor, transpose_amount):
      """Augment a NoteSequence by time stretch and pitch transposition."""
      augmented_ns = sequences_lib.stretch_note_sequence(
          ns, stretch_factor, in_place=False)
      try:
        _, num_deleted_notes = sequences_lib.transpose_note_sequence(
            augmented_ns, transpose_amount,
            min_allowed_pitch=MIN_PITCH, max_allowed_pitch=MAX_PITCH,
            in_place=True)
      except chord_symbols_lib.ChordSymbolError:
        raise datagen_beam.DataAugmentationError(
            'Transposition of chord symbol(s) failed.')
      if num_deleted_notes:
        raise datagen_beam.DataAugmentationError(
            'Transposition caused out-of-range pitch(es).')
      return augmented_ns

    augment_params = itertools.product(
        self.stretch_factors, self.transpose_amounts)
    augment_fns = [
        functools.partial(augment_note_sequence,
                          stretch_factor=s, transpose_amount=t)
        for s, t in augment_params
    ]

    datagen_beam.generate_conditional_examples(
        input_transform=self.performances_input_transform(tmp_dir),
        output_dir=data_dir,
        problem_name=self.dataset_filename(),
        splits=self.splits,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        melody=False,
        noisy=False,
        encode_performance_fn=self.performance_encoder().encode_note_sequence,
        encode_score_fns=dict((name, encoder.encode_note_sequence)
                              for name, encoder in self.score_encoders()),
        augment_fns=augment_fns,
        num_replications=self.num_replications)

  def example_reading_spec(self):
    data_fields = {
        'inputs': tf.VarLenFeature(tf.int64),
        'targets': tf.VarLenFeature(tf.int64)
    }
    for name, _ in self.score_encoders():
      data_fields[name] = tf.VarLenFeature(tf.int64)

    # We don't actually "decode" anything here; the encodings are simply read as
    # tensors.
    data_items_to_decoders = None

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    return problem.preprocess_example_common(example, mode, hparams)


class ConditionalMelodyScore2PerfProblem(Score2PerfProblem):
  """Lightweight version of base class for score-to-performance problems.

  This version incorporates one performance conditioning signal.
  Data files contain tf.Example protos with encoded performance in 'targets' and
  encoded score in 'melody' and 'performance'.
  """

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    del task_id

    from magenta.models.score2perf import datagen_beam  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    def augment_note_sequence(ns, stretch_factor, transpose_amount):
      """Augment a NoteSequence by time stretch and pitch transposition."""
      augmented_ns = sequences_lib.stretch_note_sequence(
          ns, stretch_factor, in_place=False)
      try:
        _, num_deleted_notes = sequences_lib.transpose_note_sequence(
            augmented_ns, transpose_amount,
            min_allowed_pitch=MIN_PITCH, max_allowed_pitch=MAX_PITCH,
            in_place=True)
      except chord_symbols_lib.ChordSymbolError:
        raise datagen_beam.DataAugmentationError(
            'Transposition of chord symbol(s) failed.')
      if num_deleted_notes:
        raise datagen_beam.DataAugmentationError(
            'Transposition caused out-of-range pitch(es).')
      return augmented_ns

    augment_params = itertools.product(
        self.stretch_factors, self.transpose_amounts)
    augment_fns = [
        functools.partial(augment_note_sequence,
                          stretch_factor=s, transpose_amount=t)
        for s, t in augment_params
    ]
    datagen_beam.generate_conditional_examples(
        input_transform=self.performances_input_transform(tmp_dir),
        output_dir=data_dir,
        problem_name=self.dataset_filename(),
        splits=self.splits,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        melody=True,
        noisy=False,
        encode_performance_fn=self.performance_encoder().encode_note_sequence,
        encode_score_fns=dict((name, encoder.encode_note_sequence)
                              for name, encoder in self.score_encoders()),
        augment_fns=augment_fns,
        num_replications=self.num_replications)

  def hparams(self, defaults, model_hparams):
    del model_hparams   # unused
    perf_encoder = self.get_feature_encoders()['targets']
    defaults.modality = {'targets': t2t_modalities.ModalityType.SYMBOL}
    defaults.vocab_size = {'targets': perf_encoder.vocab_size}
    if self.has_inputs:
      score_encoder = self.score_encoders()
      # iterate over each score encoder and update modality/vocab_size
      for name, se in score_encoder:
        defaults.modality[name] = t2t_modalities.ModalityType.SYMBOL
        defaults.vocab_size[name] = se.vocab_size

  def feature_encoders(self, data_dir):
    del data_dir
    encoders = {
        'targets': self.performance_encoder()
    }
    score_encoders = self.score_encoders()
    # CompositeScoreEncoder is tricky, so using a list of encoders instead.
    if len(score_encoders) > 1:
      for name, encoder in score_encoders:
        encoders[name] = encoder
    else:
      # If only one score component, just use its encoder.
      _, encoders['inputs'] = score_encoders[0]
    return encoders

  def example_reading_spec(self):
    data_fields = {
        'targets': tf.VarLenFeature(tf.int64),
    }
    for name, _ in self.score_encoders():
      data_fields[name] = tf.VarLenFeature(tf.int64)

    # We don't actually "decode" anything here; the encodings are simply read as
    # tensors.
    data_items_to_decoders = None

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    return problem.preprocess_example_common(example, mode, hparams)


class ConditionalMelodyNoisyScore2PerfProblem(
    ConditionalMelodyScore2PerfProblem):
  """Lightweight version of base class for score-to-performance problems.

  This version incorporates one performance conditioning signal.
  Data files contain tf.Example protos with encoded performance in 'targets' and
  encoded score in 'melody' and 'performance'.
  """

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    del task_id

    from magenta.models.score2perf import datagen_beam  # pylint: disable=g-import-not-at-top,import-outside-toplevel

    def augment_note_sequence(ns, stretch_factor, transpose_amount):
      """Augment a NoteSequence by time stretch and pitch transposition."""
      augmented_ns = sequences_lib.stretch_note_sequence(
          ns, stretch_factor, in_place=False)
      try:
        _, num_deleted_notes = sequences_lib.transpose_note_sequence(
            augmented_ns, transpose_amount,
            min_allowed_pitch=MIN_PITCH, max_allowed_pitch=MAX_PITCH,
            in_place=True)
      except chord_symbols_lib.ChordSymbolError:
        raise datagen_beam.DataAugmentationError(
            'Transposition of chord symbol(s) failed.')
      if num_deleted_notes:
        raise datagen_beam.DataAugmentationError(
            'Transposition caused out-of-range pitch(es).')
      return augmented_ns

    augment_params = itertools.product(
        self.stretch_factors, self.transpose_amounts)
    augment_fns = [
        functools.partial(augment_note_sequence,
                          stretch_factor=s, transpose_amount=t)
        for s, t in augment_params
    ]
    datagen_beam.generate_conditional_examples(
        input_transform=self.performances_input_transform(tmp_dir),
        output_dir=data_dir,
        problem_name=self.dataset_filename(),
        splits=self.splits,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        melody=True,
        noisy=True,
        encode_performance_fn=self.performance_encoder().encode_note_sequence,
        encode_score_fns=dict((name, encoder.encode_note_sequence)
                              for name, encoder in self.score_encoders()),
        augment_fns=augment_fns,
        num_replications=self.num_replications)


class Chords2PerfProblem(Score2PerfProblem):
  """Base class for musical chords-to-performance problems."""

  def score_encoders(self):
    return [('chords', music_encoders.TextChordsEncoder(steps_per_quarter=1))]


class Melody2PerfProblem(Score2PerfProblem):
  """Base class for musical melody-to-performance problems."""

  def score_encoders(self):
    return [
        ('melody', music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


class AbsoluteMelody2PerfProblem(Score2PerfProblem):
  """Base class for musical (absolute-timed) melody-to-performance problems."""

  @property
  def absolute_timing(self):
    return True

  def score_encoders(self):
    return [
        ('melody', music_encoders.TextMelodyEncoderAbsolute(
            steps_per_second=10, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


class LeadSheet2PerfProblem(Score2PerfProblem):
  """Base class for musical lead-sheet-to-performance problems."""

  def score_encoders(self):
    return [
        ('chords', music_encoders.TextChordsEncoder(steps_per_quarter=4)),
        ('melody', music_encoders.TextMelodyEncoder(
            steps_per_quarter=4, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH))
    ]


@registry.register_problem('score2perf_maestro_language_uncropped_aug')
class Score2PerfMaestroLanguageUncroppedAug(Score2PerfProblem):
  """Piano performance language model on the MAESTRO dataset."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return _maestro_input_transform()

  @property
  def splits(self):
    return None

  @property
  def min_hop_size_seconds(self):
    return 0.0

  @property
  def max_hop_size_seconds(self):
    return 0.0

  @property
  def add_eos_symbol(self):
    return False

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]

  @property
  def random_crop_in_train(self):
    return True

  @property
  def split_in_eval(self):
    return True


@registry.register_problem('score2perf_maestro_absmel2perf_5s_to_30s_aug10x')
class Score2PerfMaestroAbsMel2Perf5sTo30sAug10x(AbsoluteMelody2PerfProblem):
  """Generate performances from an absolute-timed melody, with augmentation."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return _maestro_input_transform()

  @property
  def splits(self):
    return None

  @property
  def min_hop_size_seconds(self):
    return 5.0

  @property
  def max_hop_size_seconds(self):
    return 30.0

  @property
  def num_replications(self):
    return 10

  @property
  def add_eos_symbol(self):
    return True

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]


@registry.register_problem('score2perf_maestro_perf_conditional_aug_10x')
class Score2PerfMaestroPerfConditionalAug10x(ConditionalScore2PerfProblem):
  """Generate performances from scratch (or from primer)."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return _maestro_input_transform()

  @property
  def splits(self):
    return

  @property
  def num_replications(self):
    return 10

  @property
  def add_eos_symbol(self):
    return False

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]

  @property
  def has_inputs(self):
    encoders = self.get_feature_encoders()
    return ('performance' in encoders) or ('inputs' in encoders)

  def score_encoders(self):
    return [
        ('performance', music_encoders.MidiPerformanceEncoder(
            steps_per_second=100,
            num_velocity_bins=32,
            min_pitch=21,
            max_pitch=108,
            add_eos=self.add_eos_symbol))
    ]


@registry.register_problem('score2perf_maestro_mel_perf_conditional_aug_10x')
class Score2PerfMaestroMelPerfConditionalAug10x(
    ConditionalMelodyScore2PerfProblem):
  """Generate performances from scratch (or from primer)."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return _maestro_input_transform()

  @property
  def splits(self):
    return

  @property
  def num_replications(self):
    return 10

  @property
  def add_eos_symbol(self):
    return False

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]

  @property
  def has_inputs(self):
    encoders = self.get_feature_encoders()
    return ('performance' in encoders) or ('inputs' in encoders)

  def score_encoders(self):
    return [
        ('performance', music_encoders.MidiPerformanceEncoder(
            steps_per_second=100,
            num_velocity_bins=32,
            min_pitch=21,
            max_pitch=108,
            add_eos=self.add_eos_symbol)),
        ('melody', music_encoders.TextMelodyEncoderAbsolute(
            steps_per_second=10, min_pitch=21, max_pitch=108))
    ]


@registry.register_problem('score2perf_maestro_mel_perf_conditional_noisy_10x')
class Score2PerfMaestroMelPerfConditionalNoisy10x(
    ConditionalMelodyNoisyScore2PerfProblem):
  """Generate performances from scratch (or from primer)."""

  def performances_input_transform(self, tmp_dir):
    del tmp_dir
    return _maestro_input_transform()

  @property
  def splits(self):
    return

  @property
  def num_replications(self):
    return 10

  @property
  def add_eos_symbol(self):
    return False

  @property
  def stretch_factors(self):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    return [0.95, 0.975, 1.0, 1.025, 1.05]

  @property
  def transpose_amounts(self):
    # Transpose no more than a minor third.
    return [-3, -2, -1, 0, 1, 2, 3]

  @property
  def has_inputs(self):
    encoders = self.get_feature_encoders()
    return ('performance' in encoders) or ('inputs' in encoders)

  def score_encoders(self):
    return [
        ('performance', music_encoders.MidiPerformanceEncoder(
            steps_per_second=100,
            num_velocity_bins=32,
            min_pitch=21,
            max_pitch=108,
            add_eos=self.add_eos_symbol)),
        ('melody', music_encoders.TextMelodyEncoderAbsolute(
            steps_per_second=10, min_pitch=21, max_pitch=108))
    ]


@registry.register_hparams
def score2perf_transformer_base():
  hparams = transformer.transformer_base()
  hparams.bottom['inputs'] = modalities.bottom
  return hparams
