"""Glyph classification unsupervised pipeline.

Extracts patches from stafflines, and runs k-means on the patches. Each patch
will be labeled and used for k-nearest-neighbors classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random
import shutil
import tempfile

# internal imports
from absl import flags
import apache_beam as beam
from apache_beam.transforms import combiners
import tensorflow as tf

from magenta.models.omr.training.clustering import staffline_patches_dofn

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('music_pattern', [],
                          'Pattern for the input music score PNGs.')
flags.DEFINE_string('output_path', None, 'Path to the output TFRecords.')
flags.DEFINE_integer('patch_height', 18,
                     'The normalized height of a staffline.')
flags.DEFINE_integer('patch_width', 15,
                     'The width of a horizontal patch of a staffline.')
flags.DEFINE_integer('num_stafflines', 19,
                     'The number of stafflines to extract.')
flags.DEFINE_integer('num_pages', 0, 'Subsample the pages to run on.')
flags.DEFINE_integer('num_outputs', 0, 'Number of output patches.')
flags.DEFINE_integer('max_patches_per_page', 10,
                     'Sample patches per page if above this amount.')
flags.DEFINE_integer('timeout_ms', 600000, 'Timeout for processing a page.')
flags.DEFINE_integer('kmeans_num_clusters', 1000, 'Number of k-means clusters.')
flags.DEFINE_integer('kmeans_batch_size', 10000,
                     'Batch size for mini-batch k-means.')
flags.DEFINE_integer('kmeans_num_steps', 100,
                     'Number of k-means training steps.')


def train_kmeans(patch_file_pattern, num_clusters, batch_size, train_steps):
  """Runs TensorFlow K-Means over TFRecords.

  Args:
    patch_file_pattern: Pattern that matches TFRecord file(s) holding Examples
        with image patches.
    num_clusters: Number of output clusters.
    batch_size: Size of a k-means minibatch.
    train_steps: Number of steps for k-means training.

  Returns:
    A NumPy array of shape (num_clusters, patch_height * patch_width). The
        cluster centers.
  """

  def input_fn():
    """The tf.learn input_fn.

    Returns:
      features, a float32 tensor of shape
          (batch_size, patch_height * patch_width).
      None for labels (not applicable to k-means).
    """
    examples = tf.contrib.learn.read_batch_examples(
        patch_file_pattern,
        batch_size,
        tf.TFRecordReader,
        queue_capacity=batch_size * 2)
    features = tf.parse_example(examples, {
        'features':
            tf.FixedLenFeature(FLAGS.patch_height * FLAGS.patch_width,
                               tf.float32)
    })['features']
    return features, None  # no labels

  def experiment_fn(run_config, unused_hparams):
    """The tf.learn experiment_fn.

    Args:
      run_config: The run config to be passed to the KMeansClustering.
      unused_hparams: Hyperparameters; not applicable.

    Returns:
      A tf.contrib.learn.Experiment.
    """
    kmeans = tf.contrib.learn.KMeansClustering(
        num_clusters=num_clusters, config=run_config)
    return tf.contrib.learn.Experiment(
        estimator=kmeans,
        train_steps=train_steps,
        train_input_fn=input_fn,
        eval_steps=1,
        eval_input_fn=input_fn)

  output_dir = tempfile.mkdtemp(prefix='staffline_patches_kmeans')
  try:
    learn_runner.run(
        experiment_fn,
        run_config=tf.contrib.learn.RunConfig(model_dir=output_dir))
    num_features = FLAGS.patch_height * FLAGS.patch_width
    clusters_t = tf.Variable(
        tf.zeros((num_clusters, num_features)),  # Dummy init op
        name='clusters')
    with tf.Session() as sess:
      tf.train.Saver(var_list=[clusters_t]).restore(
          sess, os.path.join(output_dir, 'model.ckpt-%d' % train_steps))
      return clusters_t.eval()
  finally:
    shutil.rmtree(output_dir)


def main(_):
  tf.logging.info('Building the pipeline...')
  records_dir = tempfile.mkdtemp(prefix='staffline_kmeans')
  try:
    patch_file_prefix = os.path.join(records_dir, 'patches')
    with beam.Pipeline() as pipeline:
      filenames = file_io.get_matching_files(FLAGS.music_pattern)
      assert filenames, 'Must have matched some filenames'
      if 0 < FLAGS.num_pages < len(filenames):
        filenames = random.sample(filenames, FLAGS.num_pages)
      filenames = pipeline | beam.transforms.Create(filenames)
      patches = filenames | beam.ParDo(
          staffline_patches_dofn.StafflinePatchesDoFn(
              patch_height=FLAGS.patch_height,
              patch_width=FLAGS.patch_width,
              num_stafflines=FLAGS.num_stafflines,
              timeout_ms=FLAGS.timeout_ms,
              max_patches_per_page=FLAGS.max_patches_per_page))
      if FLAGS.num_outputs:
        patches |= combiners.Sample.FixedSizeGlobally(FLAGS.num_outputs)
      patches |= beam.io.WriteToTFRecord(
          patch_file_prefix, beam.coders.ProtoCoder(tf.train.Example))
      tf.logging.info('Running the pipeline...')
    tf.logging.info('Running k-means...')
    patch_files = file_io.get_matching_files(patch_file_prefix + '*')
    clusters = train_kmeans(patch_files, FLAGS.kmeans_num_clusters,
                            FLAGS.kmeans_batch_size, FLAGS.kmeans_num_steps)
    tf.logging.info('Writing the centroids...')
    with tf_record.TFRecordWriter(FLAGS.output_path) as writer:
      for cluster in clusters:
        example = tf.train.Example()
        example.features.feature['features'].float_list.value.extend(cluster)
        example.features.feature['height'].int64_list.value.append(
            FLAGS.patch_height)
        example.features.feature['width'].int64_list.value.append(
            FLAGS.patch_width)
        writer.write(example.SerializeToString())
    tf.logging.info('Done!')
  finally:
    shutil.rmtree(records_dir)


if __name__ == '__main__':
  tf.app.run(main)
