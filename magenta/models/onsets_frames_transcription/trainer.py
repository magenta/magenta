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

r"""Train Onsets and Frames piano transcription model."""

from __future__ import absolute_import, division, print_function

import functools
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotmap import DotMap

import tensorflow.compat.v1 as tf

tf.enable_v2_behavior()
tf.enable_eager_execution()

tf.autograph.set_verbosity(0)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('using_plaidml', False, 'Are we using plaidml')

tf.app.flags.DEFINE_string('model_id', None, 'Id to save the model as')
tf.app.flags.DEFINE_string('load_id', '*', 'Id of the model to load')

tf.app.flags.DEFINE_string('master', '',
                           'Name of the TensorFlow runtime to use.')
tf.app.flags.DEFINE_string('config', 'onsets_frames',
                           'Name of the config to use.')
tf.app.flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of train/eval examples.')
tf.app.flags.DEFINE_boolean(
    'preprocess_examples', True,
    'Whether to preprocess examples or assume they have already been '
    'preprocessed.')
tf.app.flags.DEFINE_string(
    'model_dir', '~/tmp/onsets_frames',
    'Path where checkpoints and summary events will be located during '
    'training and evaluation.')
tf.app.flags.DEFINE_string('eval_name', None, 'Name for this eval run.')
tf.app.flags.DEFINE_integer('num_steps', 1000000,
                            'Number of training steps or `None` for infinite.')
tf.app.flags.DEFINE_integer(
    'eval_num_steps', None,
    'Number of eval steps or `None` to go through all examples.')
tf.app.flags.DEFINE_integer(
    'keep_checkpoint_max', 100,
    'Maximum number of checkpoints to keep in `train` mode or 0 for infinite.')
tf.app.flags.DEFINE_string(
    'hparams', '{}',
    'Json of `name: value` hyperparameter values. '
    'ex. --hparams={\"frames_true_weighing\":2,\"onsets_true_weighing\":8}')
tf.app.flags.DEFINE_boolean('use_tpu', False,
                            'Whether training will happen on a TPU.')
tf.app.flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'predict'],
                         'Which mode to use.')
tf.app.flags.DEFINE_string(
    'log', 'ERROR',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_string(
    'dataset_source', 'MAESTRO',
    'Source of the dataset (MAESTRO, NSYNTH)'
)
tf.app.flags.DEFINE_string(
    'audio_filename', '',
    'Audio file to transcribe'
)
tf.app.flags.DEFINE_enum('model_type', 'MIDI', ['MIDI', 'TIMBRE', 'FULL'],
                         'type of model to train')

tf.app.flags.DEFINE_enum('dataset_name', 'nsynth', ['nsynth', 'slakh'],
                         'type of dataset we are using')

tf.app.flags.DEFINE_string(
    'transcribed_file_suffix', 'predicted',
    'Optional suffix to add to transcribed files.')

if FLAGS.using_plaidml:
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    import plaidml.keras

    plaidml.keras.install_backend()

from magenta.models.onsets_frames_transcription import data, train_util, configs, timbre_dataset_reader, \
    model_util


def run(config_map, data_fn, additional_trial_info):
    """Run training or evaluation."""
    tf.logging.set_verbosity(FLAGS.log)

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
            master=FLAGS.master,
            model_dir=model_dir,
            name=FLAGS.eval_name,
            preprocess_examples=FLAGS.preprocess_examples,
            hparams=hparams,
            num_steps=FLAGS.eval_num_steps)
    else:
        raise ValueError('Unknown/unsupported mode: %s' % FLAGS.mode)


def main(argv):
    del argv
    tf.app.flags.mark_flags_as_required(['examples_path'])
    provide_batch_fn = timbre_dataset_reader.provide_batch if model_util.ModelType[
                                                 FLAGS.model_type] == model_util.ModelType.TIMBRE \
        else data.provide_batch
    data_fn = functools.partial(provide_batch_fn,
                                examples=FLAGS.examples_path,
                                dataset_name=FLAGS.dataset_name) \
        if FLAGS.examples_path else None
    additional_trial_info = {'examples_path': FLAGS.examples_path}
    run(config_map=configs.CONFIG_MAP, data_fn=data_fn,
        additional_trial_info=additional_trial_info)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
