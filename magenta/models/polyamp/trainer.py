# Copyright 2020 The Magenta Authors.
# Modifications Copyright 2020 Jack Spencer Smith.
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

r"""Train Onsets and Frames piano transcription model."""

import functools
import json
import os

import absl.flags
import tensorflow as tf
from absl import app, logging
from dotmap import DotMap

from magenta.models.polyamp import configs, dataset_reader, model_util, nsynth_dataset_reader, \
    slakh_dataset_reader, train_util
from magenta.models.polyamp.dataset_reader import merge_data_functions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.autograph.set_verbosity(0)

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_boolean('using_plaidml', False, 'Are we using plaidml')

absl.flags.DEFINE_string('model_id', None, 'Id to save the model as')
absl.flags.DEFINE_string('load_id', '*', 'Id of the model to load')

absl.flags.DEFINE_string('master', '',
                         'Name of the TensorFlow runtime to use.')
absl.flags.DEFINE_string('config', 'onsets_frames',
                         'Name of the config to use.')
absl.flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of train/eval examples.')
absl.flags.DEFINE_string(
    'nsynth_examples_path', None,
    'Optional path to a TFRecord for full model training on nsynth-style data.'
)
absl.flags.DEFINE_boolean(
    'preprocess_examples', True,
    'Whether to preprocess examples or assume they have already been '
    'preprocessed.')
absl.flags.DEFINE_string(
    'model_dir', '~/tmp/onsets_frames',
    'Path where checkpoints and summary events will be located during '
    'training and evaluation.')
absl.flags.DEFINE_string('eval_name', None, 'Name for this eval run.')
absl.flags.DEFINE_integer('num_steps', 1000000,
                          'Number of training steps or `None` for infinite.')
absl.flags.DEFINE_integer(
    'eval_num_steps', None,
    'Number of eval steps or `None` to go through all examples.')
absl.flags.DEFINE_boolean('note_based', False,
                          'Whether eval metrics are note-based for Melodic model.')
absl.flags.DEFINE_integer(
    'keep_checkpoint_max', 100,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
absl.flags.DEFINE_string(
    'hparams', '{}',
    'Json of `name: value` hyperparameter values. '
    'ex. --hparams={\"frames_true_weighing\":2,\"onsets_true_weighing\":8}')
absl.flags.DEFINE_boolean('use_tpu', False,
                          'Whether training will happen on a TPU.')
absl.flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'predict'],
                       'Which mode to use.')
absl.flags.DEFINE_string(
    'log', 'ERROR',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
absl.flags.DEFINE_string(
    'dataset_source', 'MAESTRO',
    'Source of the dataset (MAESTRO, NSYNTH)'
)
absl.flags.DEFINE_string(
    'audio_filename', '',
    'Audio file to transcribe'
)
absl.flags.DEFINE_enum('model_type', 'MIDI', ['MIDI', 'TIMBRE', 'FULL'],
                       'type of model to train')

absl.flags.DEFINE_enum('dataset_name', 'nsynth', ['nsynth', 'slakh'],
                       'type of dataset we are using')

absl.flags.DEFINE_string(
    'transcribed_file_suffix', 'predicted',
    'Optional suffix to add to transcribed files.')


def run(config_map, data_fn, additional_trial_info):
    """Run training or evaluation."""
    logging.set_verbosity(FLAGS.log)

    config = config_map[FLAGS.config]
    model_dir = os.path.expanduser(FLAGS.model_dir)

    hparams = config.hparams

    # Command line flags override any of the preceding hyperparameter values.
    hparams.update(json.loads(FLAGS.hparams))
    hparams = DotMap(hparams)

    hparams.using_plaidml = FLAGS.using_plaidml
    hparams.model_id = FLAGS.model_id
    hparams.load_id = FLAGS.load_id

    hparams.model_type = model_util.ModelType[FLAGS.model_type]
    hparams.split_pianoroll = model_util.ModelType[FLAGS.model_type] is model_util.ModelType.FULL

    if FLAGS.mode == 'train':
        train_util.train(
            data_fn=data_fn,
            model_dir=model_dir,
            model_type=model_util.ModelType[FLAGS.model_type],
            preprocess_examples=FLAGS.preprocess_examples,
            hparams=hparams,
            num_steps=FLAGS.num_steps)
    elif FLAGS.mode == 'predict':
        train_util.transcribe(data_fn=data_fn,
                              model_dir=model_dir,
                              model_type=model_util.ModelType[FLAGS.model_type],
                              path=FLAGS.audio_filename,
                              file_suffix=FLAGS.transcribed_file_suffix,
                              hparams=hparams
                              )
    elif FLAGS.mode == 'eval':
        train_util.evaluate(
            # model_fn=config.model_fn,
            data_fn=data_fn,
            additional_trial_info=additional_trial_info,
            model_dir=model_dir,
            model_type=model_util.ModelType[FLAGS.model_type],
            name=FLAGS.eval_name,
            preprocess_examples=FLAGS.preprocess_examples,
            hparams=hparams,
            num_steps=FLAGS.eval_num_steps,
            note_based=FLAGS.note_based)
    else:
        raise ValueError('Unknown/unsupported mode: %s' % FLAGS.mode)


def main(argv):
    del argv
    absl.flags.mark_flags_as_required(['examples_path'])
    if model_util.ModelType[FLAGS.model_type] is model_util.ModelType.TIMBRE:
        provide_batch_fn = (nsynth_dataset_reader.provide_batch
                            if FLAGS.dataset_name == 'nsynth'
                            else slakh_dataset_reader.provide_batch)
    else:
        provide_batch_fn = dataset_reader.provide_batch

    data_fn = None
    if FLAGS.examples_path:
        data_fn = functools.partial(provide_batch_fn,
                                    examples=FLAGS.examples_path)
    if (FLAGS.nsynth_examples_path
            and model_util.ModelType[FLAGS.model_type] is model_util.ModelType.FULL):
        nsynth_fn = functools.partial(nsynth_dataset_reader.provide_batch,
                                      examples=FLAGS.nsynth_examples_path,
                                      for_full_model=True)
        data_fn = (merge_data_functions([nsynth_fn, data_fn])
                   if data_fn is not None else nsynth_fn)
    additional_trial_info = {'examples_path': FLAGS.examples_path}
    run(config_map=configs.CONFIG_MAP, data_fn=data_fn,
        additional_trial_info=additional_trial_info)


def console_entry_point():
    app.run(main)


if __name__ == '__main__':
    console_entry_point()
