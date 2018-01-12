"""Script to evaluate a dataset fold under a model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# internal imports
import numpy as np
import tensorflow as tf
from magenta.models.coconet import lib_data
from magenta.models.coconet import lib_evaluation
from magenta.models.coconet import lib_graph
from magenta.models.coconet import lib_util

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_string('data_dir', None,
                    'Path to the base directory for different datasets.')
flags.DEFINE_string('eval_logdir', None,
                    'Path to the base directory for saving evaluation '
                    'statistics.')
flags.DEFINE_string('fold', None,
                    'Data fold on which to evaluate (valid or test)')
flags.DEFINE_string('fold_index', None,
                    'Optionally, index of particular data point in fold to '
                    'evaluate.')
flags.DEFINE_string('unit', None, 'Note or frame or example.')
flags.DEFINE_integer('ensemble_size', 5,
                     'Number of ensemble members to average.')
flags.DEFINE_bool('chronological', False,
                  'Indicates evaluation should proceed in chronological order.')
flags.DEFINE_string('checkpoint', None, 'Path to checkpoint directory.')
flags.DEFINE_string('sample_npy_path', None, 'Path to samples to be evaluated.')


def main(unused_argv):
  if FLAGS.checkpoint is None or not FLAGS.checkpoint:
    raise ValueError(
        'Need to provide a path to checkpoint directory.')
  wmodel = lib_graph.load_checkpoint(FLAGS.checkpoint)
  if FLAGS.eval_logdir is None:
    raise ValueError(
        'Set flag eval_logdir to specify a path for saving eval statistics.')
  else:
    eval_logdir = os.path.join(FLAGS.eval_logdir, 'eval_stats')
    tf.gfile.MakeDirs(eval_logdir)

  evaluator = lib_evaluation.BaseEvaluator.make(
      FLAGS.unit, wmodel=wmodel, chronological=FLAGS.chronological)
  evaluator = lib_evaluation.EnsemblingEvaluator(evaluator, FLAGS.ensemble_size)

  if not FLAGS.sample_npy_path and FLAGS.fold is None:
    raise ValueError(
        'Either --fold must be specified, or paths of npy files to load must '
        'be given, but not both.')
  if FLAGS.fold is not None:
    evaluate_fold(FLAGS.fold, evaluator, wmodel.hparams, eval_logdir)
  if FLAGS.sample_npy_path is not None:
    evaluate_paths([FLAGS.sample_npy_path], evaluator, wmodel.hparams,
                   eval_logdir)
  print('Done')


def evaluate_fold(fold, evaluator, hparams, eval_logdir):
  """Writes to file the neg. loglikelihood of given fold (train/valid/test)."""
  eval_run_name = 'eval_%s_%s%s_%s_ensemble%s_chrono%s' % (
      lib_util.timestamp(), fold, FLAGS.fold_index
      if FLAGS.fold_index is not None else '', FLAGS.unit, FLAGS.ensemble_size,
      FLAGS.chronological)
  log_fname = '%s__%s.npz' % (os.path.basename(FLAGS.checkpoint), eval_run_name)
  log_fpath = os.path.join(eval_logdir, log_fname)

  pianorolls = get_fold_pianorolls(fold, hparams)

  rval = lib_evaluation.evaluate(evaluator, pianorolls)
  print('Writing to path: %s' % log_fpath)
  with lib_util.atomic_file(log_fpath) as p:
    np.savez_compressed(p, **rval)


def evaluate_paths(paths, evaluator, unused_hparams, eval_logdir):
  """Evaluates negative loglikelihood of pianorolls from given paths."""
  for path in paths:
    name = 'eval_samples_%s_%s_ensemble%s_chrono%s' % (lib_util.timestamp(),
                                                       FLAGS.unit,
                                                       FLAGS.ensemble_size,
                                                       FLAGS.chronological)
    log_fname = '%s__%s.npz' % (os.path.splitext(os.path.basename(path))[0],
                                name)
    log_fpath = os.path.join(eval_logdir, log_fname)

    pianorolls = get_path_pianorolls(path)
    rval = lib_evaluation.evaluate(evaluator, pianorolls)
    print('Writing evaluation statistics to', log_fpath)
    with lib_util.atomic_file(log_fpath) as p:
      np.savez_compressed(p, **rval)


def get_fold_pianorolls(fold, hparams):
  dataset = lib_data.get_dataset(FLAGS.data_dir, hparams, fold)
  pianorolls = dataset.get_pianorolls()
  print('\nRetrieving pianorolls from %s set of %s dataset.\n' %
        (fold, hparams.dataset))
  print_statistics(pianorolls)
  if FLAGS.fold_index is not None:
    pianorolls = [pianorolls[int(FLAGS.fold_index)]]
  return pianorolls


def get_path_pianorolls(path):
  pianoroll_fpath = os.path.join(tf.resource_loader.get_data_files_path(), path)
  print('Retrieving pianorolls from', pianoroll_fpath)
  with tf.gfile.Open(pianoroll_fpath, 'r') as p:
    pianorolls = np.load(p)
  if isinstance(pianorolls, np.ndarray):
    print(pianorolls.shape)
  print_statistics(pianorolls)
  return pianorolls


def print_statistics(pianorolls):
  """Prints statistics of given pianorolls, such as max and unique length."""
  if isinstance(pianorolls, np.ndarray):
    print(pianorolls.shape)
  print('# of total pieces in set:', len(pianorolls))
  lengths = [len(roll) for roll in pianorolls]
  if len(np.unique(lengths)) > 1:
    print('lengths', np.sort(lengths))
  print('max_len', max(lengths))
  print('unique lengths',
        np.unique(sorted(pianoroll.shape[0] for pianoroll in pianorolls)))
  print('shape', pianorolls[0].shape)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
